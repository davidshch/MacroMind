import praw
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from ..config import get_settings
from .base_sentiment import BaseSentimentAnalyzer
from collections import defaultdict
import re
import asyncio
import time
import json  # Added for potential serialization if caching complex objects
import hashlib  # Added hashlib
from sqlalchemy.orm import Session  # Added Session
from fastapi import Depends  # Added Depends

from ..database.models import RawSentimentAnalysis  # Added RawSentimentAnalysis model
from ..database.database import get_db  # Added get_db

logger = logging.getLogger(__name__)
settings = get_settings()

# Simple in-memory cache dictionary
_sentiment_cache = {}
_cache_expiry_seconds = 3600  # 1 hour

class SocialSentimentService(BaseSentimentAnalyzer):
    def __init__(self, db: Session = Depends(get_db)):
        super().__init__()
        self.db = db  # Store DB session
        self.reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent="MacroMind/1.0"
        )
        # Updated subreddits list with categories
        self.subreddit_weights = {
            # Stock-focused (higher weights)
            "stocks": 1.0,
            "investing": 1.0,
            "stockmarket": 1.0,
            "SecurityAnalysis": 1.0,
            "dividends": 0.9,
            "wallstreetbets": 0.7,  # Lower weight due to more noise
            "options": 0.8,
            "pennystocks": 0.6,  # Lower weight due to higher volatility
            
            # Crypto-focused
            "cryptocurrency": 0.9,
            "cryptomarkets": 0.8,
            "bitcoin": 1.0,
            "ethereum": 1.0,
            "altcoin": 0.7  # Lower weight due to higher speculation
        }
        
        self.timeframes = {
            "24h": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }
        
        # Enhanced spam patterns
        self.spam_patterns = [
            r'to\s*the\s*moon',
            r'rockets*\s*emoji',
            r'buy\s*now',
            r'sell\s*now',
            r'price\s*target\s*\d+',
            r'pump',
            r'dump',
            r'🚀+',
            r'short\s*squeeze',
            r'manipulation',
            r'scam',
            r'ponzi',
            r'guarantee',
            r'amazing\s*opportunity',
            r'cant\s*lose',
            r'free\s*money'
        ]
        
        # Minimum content requirements
        self.min_text_length = 20
        self.min_karma_threshold = 50
        self.min_account_age_days = 30
        self.request_delay = 2  # Seconds between requests
        self.last_request_time = 0
        self.timeout = 10  # Seconds

    def _generate_content_hash(self, text_content: str) -> str:
        """Generates an SHA256 hash for a given text content."""
        return hashlib.sha256(text_content.encode('utf-8')).hexdigest()

    async def _store_raw_reddit_analysis(
        self, 
        symbol: Optional[str], 
        text_content: str, 
        source_type: str,  # e.g., "Reddit_Post", "Reddit_Comment"
        source_id: str,  # e.g., post.id or comment.id
        sentiment_result: Dict[str, Any],
        created_utc: datetime,
        subreddit_name: Optional[str] = None
    ):
        """Stores the raw sentiment analysis of a single Reddit item into the database."""
        if "error" in sentiment_result or not sentiment_result.get("all_scores"):
            logger.warning(f"Skipping DB storage for {source_type} id {source_id} due to sentiment analysis error or missing scores.")
            return

        content_hash = self._generate_content_hash(text_content)
        source_identifier = f"{source_type}_{subreddit_name}_{source_id}" if subreddit_name else f"{source_type}_{source_id}"

        try:
            loop = asyncio.get_event_loop()
            existing_analysis = await loop.run_in_executor(None,
                lambda: self.db.query(RawSentimentAnalysis).filter_by(text_content_hash=content_hash).first()
            )
            if existing_analysis:
                logger.debug(f"Raw analysis for content hash {content_hash} ({source_identifier}) already exists. Skipping storage.")
                return
        except Exception as e:
            logger.error(f"Error checking for existing raw Reddit analysis (hash: {content_hash}): {e}", exc_info=True)

        raw_analysis_entry = RawSentimentAnalysis(
            symbol=symbol,
            text_content_hash=content_hash,
            text_content=text_content,
            source=source_identifier,
            sentiment_label=sentiment_result["primary_sentiment"],
            sentiment_score=sentiment_result["primary_confidence"],
            all_scores=sentiment_result["all_scores"],
            analyzed_at=datetime.utcnow(),
            source_created_at=created_utc
        )
        try:
            self.db.add(raw_analysis_entry)
            await loop.run_in_executor(None, self.db.commit)
            logger.debug(f"Stored raw Reddit analysis for: {source_identifier}")
        except Exception as e:
            logger.error(f"DB Error storing raw Reddit analysis for {source_identifier}: {e}", exc_info=True)
            await loop.run_in_executor(None, self.db.rollback)

    async def get_reddit_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get enhanced sentiment analysis from Reddit posts and comments."""
        try:
            cache_key = f"reddit_sentiment_v2:{symbol}"  # Ensure cache key is distinct
            cached_data = self._get_cached_sentiment(cache_key)
            if cached_data:
                logger.info(f"Returning cached Reddit sentiment for {symbol}")
                return cached_data

            all_timeframe_data = {}
            storage_tasks = []  # For storing raw analysis
            
            for timeframe, delta in self.timeframes.items():
                try:
                    posts = await asyncio.wait_for(
                        self._gather_posts(symbol, delta),
                        timeout=self.timeout
                    )
                    if not posts:
                        all_timeframe_data[timeframe] = self._create_empty_timeframe_sentiment(symbol)
                        continue

                    post_sentiments_for_aggregation = []
                    comment_sentiments_for_aggregation = []
                    total_engagement = 0

                    for post_data in posts:  # Renamed post to post_data to avoid conflict
                        post_text_content = f"{post_data['title']} {post_data['text']}"
                        if self._is_relevant_content(post_text_content, symbol):
                            # Analyze post sentiment using MLModelFactory via BaseSentimentAnalyzer
                            sentiment_result = await self.ml_model_factory.analyze_sentiment(post_text_content)
                            
                            if "error" not in sentiment_result:
                                storage_tasks.append(self._store_raw_reddit_analysis(
                                    symbol, post_text_content, "Reddit_Post", post_data.get('id', 'unknown_id'), 
                                    sentiment_result, post_data['created'], post_data['subreddit']
                                ))
                                weighted_sentiment = self._apply_weights(
                                    sentiment_result, post_data['score'], post_data['author_karma'],
                                    post_data['created'], post_data['subreddit']
                                )
                                post_sentiments_for_aggregation.append(weighted_sentiment)
                            total_engagement += post_data['score'] + post_data['comments']

                            for comment_data in post_data['top_comments']:  # Renamed comment to comment_data
                                if self._is_relevant_content(comment_data['text'], symbol):
                                    comment_sentiment_result = await self.ml_model_factory.analyze_sentiment(comment_data['text'])
                                    if "error" not in comment_sentiment_result:
                                        storage_tasks.append(self._store_raw_reddit_analysis(
                                            symbol, comment_data['text'], "Reddit_Comment", comment_data.get('id', 'unknown_id'),
                                            comment_sentiment_result, comment_data['created'], post_data['subreddit']
                                        ))
                                        weighted_comment_sentiment = self._apply_weights(
                                            comment_sentiment_result, comment_data['score'], comment_data['author_karma'],
                                            comment_data['created'], post_data['subreddit']
                                        )
                                        comment_sentiments_for_aggregation.append(weighted_comment_sentiment)
                                    total_engagement += comment_data['score']

                    combined_sentiments = self._combine_sentiments(post_sentiments_for_aggregation, comment_sentiments_for_aggregation)
                    all_timeframe_data[timeframe] = self._aggregate_reddit_sentiment(
                        symbol, combined_sentiments, posts, total_engagement
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Timeout while fetching Reddit data for {symbol} during timeframe {timeframe}")
                    all_timeframe_data[timeframe] = self._create_empty_timeframe_sentiment(symbol)
                except Exception as e_inner:
                    logger.error(f"Error processing timeframe {timeframe} for {symbol}: {e_inner}", exc_info=True)
                    all_timeframe_data[timeframe] = self._create_empty_timeframe_sentiment(symbol)
            
            # Wait for all raw analysis storage tasks to complete
            if storage_tasks:
                results = await asyncio.gather(*storage_tasks, return_exceptions=True)
                for i, res in enumerate(results):
                    if isinstance(res, Exception):
                        logger.error(f"Error in _store_raw_reddit_analysis task {i}: {res}")

            final_sentiment = self._create_final_sentiment(symbol, all_timeframe_data)
            self._cache_sentiment(cache_key, final_sentiment)
            return final_sentiment

        except Exception as e:
            logger.error(f"Overall Reddit sentiment analysis error for {symbol}: {str(e)}", exc_info=True)
            return self._create_final_sentiment(symbol, {tf: self._create_empty_timeframe_sentiment(symbol) for tf in self.timeframes})

    async def _process_post(self, post: Any, subreddit: str) -> Optional[Dict[str, Any]]:
        """Process a single Reddit post, include post ID."""
        try:
            author_karma = 0
            if post.author:
                author_karma = post.author.link_karma + post.author.comment_karma

            post.comments.replace_more(limit=0)
            top_comments = []
            for comment in list(post.comments)[:5]:
                comment_author_karma = 0
                if comment.author:
                    comment_author_karma = comment.author.link_karma + comment.author.comment_karma
                
                top_comments.append({
                    'id': comment.id,  # Store comment ID
                    'text': comment.body,
                    'score': comment.score,
                    'author_karma': comment_author_karma,
                    'created': datetime.utcfromtimestamp(comment.created_utc)
                })

            return {
                'id': post.id,  # Store post ID
                'title': post.title,
                'text': post.selftext,
                'score': post.score,
                'comments': post.num_comments,
                'author_karma': author_karma,
                'created': datetime.utcfromtimestamp(post.created_utc),
                'top_comments': top_comments,
                'subreddit': subreddit
            }
        except Exception as e:
            logger.warning(f"Error processing post {getattr(post, 'id', 'UNKNOWN_ID')}: {str(e)}")
            return None

    # ... (rest of the class methods remain unchanged) ...
