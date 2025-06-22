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
from sqlalchemy.ext.asyncio import AsyncSession  # Changed from Session to AsyncSession
from fastapi import Depends  # Added Depends
import os

from ..database.models import RawSentimentAnalysis  # Added RawSentimentAnalysis model
from ..database.database import get_db  # Added get_db

logger = logging.getLogger(__name__)
settings = get_settings()

# Simple in-memory cache dictionary
_sentiment_cache = {}
_cache_expiry_seconds = 3600  # 1 hour

class SocialSentimentService:
    def __init__(self, db: AsyncSession):  # Changed from Session to AsyncSession
        self.db = db  # Store DB session
        # Use composition instead of inheritance
        self.sentiment_analyzer = BaseSentimentAnalyzer()
        
        # Check if Reddit credentials are available
        if not settings.reddit_client_id or not settings.reddit_client_secret:
            logger.warning("Reddit credentials not configured. Reddit sentiment will be disabled.")
            self.reddit = None
        else:
            try:
                self.reddit = praw.Reddit(
                    client_id=settings.reddit_client_id,
                    client_secret=settings.reddit_client_secret,
                    user_agent="MacroMind/1.0"
                )
                logger.info("Reddit client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit client: {e}")
                self.reddit = None
        
        # Simplified subreddits - focus on the most reliable ones
        self.subreddit_weights = {
            "stocks": 1.0,
            "investing": 1.0,
            "stockmarket": 1.0,
            "wallstreetbets": 0.7,  # Lower weight due to noise
        }
        
        # Simplified timeframes - extend to 7 days for more data
        self.timeframes = {
            "7d": timedelta(days=7),
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
        
        # Relaxed content requirements
        self.min_text_length = 10  # Reduced from 20
        self.min_karma_threshold = 10  # Reduced from 50
        self.min_account_age_days = 7  # Reduced from 30
        self.request_delay = 1
        self.last_request_time = 0
        self.timeout = 15  # Reduced timeout for faster response

    def _generate_content_hash(self, text_content: str) -> str:
        """Generates an SHA256 hash for a given text content."""
        return hashlib.sha256(text_content.encode('utf-8')).hexdigest()

    async def _store_raw_reddit_analysis(
        self, 
        symbol: Optional[str], 
        text_content: str, 
        source_type: str,
        source_id: str,
        sentiment_result: Dict[str, Any],
        created_utc: datetime,
        subreddit_name: Optional[str] = None
    ):
        """Stores the raw sentiment analysis of a single Reddit post/comment."""
        content_hash = self._generate_content_hash(text_content)
        
        try:
            stmt = select(RawSentimentAnalysis).where(RawSentimentAnalysis.text_content_hash == content_hash)
            result = await self.db.execute(stmt)
            existing_analysis = result.scalar_one_or_none()
            if existing_analysis:
                logger.debug(f"Raw Reddit analysis for content hash {content_hash} already exists.")
                return
        except Exception as e:
            logger.error(f"Error checking for existing raw Reddit analysis: {e}", exc_info=True)
            return

        source = f"Reddit_{source_type}"
        if subreddit_name:
            source += f"_{subreddit_name}"

        raw_analysis_entry = RawSentimentAnalysis(
            symbol=symbol,
            text_content_hash=content_hash,
            text_content=text_content,
            source=source,
            sentiment_label=sentiment_result.get("primary_sentiment", "neutral"),
            sentiment_score=sentiment_result.get("primary_confidence", 0.0),
            all_scores=sentiment_result.get("all_scores"),
            analyzed_at=datetime.utcnow(),
            source_created_at=created_utc
        )
        try:
            self.db.add(raw_analysis_entry)
            logger.debug(f"Added raw Reddit analysis to session for post ID: {source_id}")
        except Exception as e:
            logger.error(f"DB Error adding raw Reddit analysis to session for post ID {source_id}: {e}", exc_info=True)
            raise

    async def get_reddit_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get Reddit sentiment for a given symbol with improved reliability."""
        # Check if Reddit client is available
        if self.reddit is None:
            logger.warning(f"Reddit client not available. Returning empty sentiment for {symbol}")
            return self._create_empty_sentiment(symbol)
            
        cache_key = f"reddit_sentiment_v3:{symbol}"
        cached_data = self._get_cached_sentiment(cache_key)
        if cached_data:
            logger.debug(f"Returning cached sentiment data for {symbol}")
            return cached_data

        try:
            # Shorter timeout for faster response
            async with asyncio.timeout(15):  # 15 second timeout
                sentiment_data = await self._gather_reddit_sentiment_fast(symbol)
                
                # Check if we have any valid data
                has_valid_data = sentiment_data.get('post_count', 0) > 0
                
                if has_valid_data:
                    logger.info(f"Successfully gathered Reddit sentiment for {symbol}: {sentiment_data.get('post_count', 0)} posts")
                    self._cache_sentiment(cache_key, sentiment_data)
                    return sentiment_data
                else:
                    logger.warning(f"No valid Reddit data found for {symbol}, returning empty structure")
                    empty_data = self._create_empty_sentiment(symbol)
                    self._cache_sentiment(cache_key, empty_data)
                    return empty_data
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout while fetching Reddit data for {symbol}")
            empty_data = self._create_empty_sentiment(symbol)
            self._cache_sentiment(cache_key, empty_data)
            return empty_data
        except Exception as e:
            logger.error(f"Error fetching Reddit sentiment for {symbol}: {e}")
            empty_data = self._create_empty_sentiment(symbol)
            self._cache_sentiment(cache_key, empty_data)
            return empty_data

    async def _process_post_simple(self, post: Any, subreddit: str) -> Optional[Dict[str, Any]]:
        """Process a single Reddit post with simplified logic."""
        try:
            author_karma = 0
            if post.author:
                author_karma = post.author.link_karma + post.author.comment_karma

            return {
                'id': post.id,
                'title': post.title,
                'text': post.selftext,
                'score': post.score,
                'comments': post.num_comments,
                'author_karma': author_karma,
                'created': datetime.utcfromtimestamp(post.created_utc),
                'subreddit': subreddit
            }
        except Exception as e:
            logger.warning(f"Error processing post {getattr(post, 'id', 'UNKNOWN_ID')}: {str(e)}")
            return None

    async def _gather_posts_fast(self, symbol: str, time_delta: timedelta) -> List[Dict[str, Any]]:
        """Gather Reddit posts for a symbol with optimized approach."""
        if self.reddit is None:
            logger.warning("Reddit client not available. Cannot gather posts.")
            return []
            
        posts = []
        cutoff_time = datetime.utcnow() - time_delta
        max_subreddits = 4  # Increased from 2 to 4 subreddits
        max_posts_per_subreddit = 5  # Increased from 3 to 5 posts per subreddit
        
        # Sort subreddits by weight to prioritize the most important ones
        sorted_subreddits = sorted(self.subreddit_weights.items(), key=lambda x: x[1], reverse=True)
        
        for subreddit_name, weight in sorted_subreddits[:max_subreddits]:
            try:
                logger.debug(f"Searching r/{subreddit_name} for {symbol}")
                subreddit = self.reddit.subreddit(subreddit_name)
                search_query = f"{symbol}"
                
                # Rate limiting
                current_time = time.time()
                if current_time - self.last_request_time < self.request_delay:
                    await asyncio.sleep(self.request_delay - (current_time - self.last_request_time))
                
                search_results = await asyncio.to_thread(
                    subreddit.search, search_query, sort='hot', limit=max_posts_per_subreddit
                )
                self.last_request_time = time.time()
                
                subreddit_posts = 0
                for post in search_results:
                    if post.created_utc and datetime.utcfromtimestamp(post.created_utc) >= cutoff_time:
                        processed_post = await self._process_post_simple(post, subreddit_name)
                        if processed_post:
                            posts.append(processed_post)
                            subreddit_posts += 1
                
                logger.debug(f"Found {subreddit_posts} posts from r/{subreddit_name} for {symbol}")
                
                # If we have enough posts, we can stop early
                if len(posts) >= 5:
                    logger.debug(f"Found sufficient posts ({len(posts)}) for {symbol}, stopping early")
                    break
                            
            except Exception as e:
                logger.warning(f"Error gathering posts from r/{subreddit_name} for {symbol}: {e}")
                continue

        logger.info(f"Gathered {len(posts)} posts for {symbol}")
        return posts

    def _is_relevant_content(self, text: str, symbol: str) -> bool:
        """Check if content is relevant and not spam."""
        if not text or len(text) < self.min_text_length:
            return False
            
        # Check for spam patterns
        text_lower = text.lower()
        for pattern in self.spam_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return False
                
        # More lenient symbol matching - check for symbol or company name variations
        symbol_variations = [
            symbol.upper(),
            symbol.lower(),
            symbol.title(),
            # Add common company name variations for popular stocks
            "nvidia" if symbol.upper() == "NVDA" else None,
            "apple" if symbol.upper() == "AAPL" else None,
            "tesla" if symbol.upper() == "TSLA" else None,
            "microsoft" if symbol.upper() == "MSFT" else None,
            "amazon" if symbol.upper() == "AMZN" else None,
        ]
        
        # Remove None values
        symbol_variations = [s for s in symbol_variations if s]
        
        # Check if any variation is mentioned
        for variation in symbol_variations:
            if variation in text_lower:
                return True
                
        return False

    def _apply_weights(self, sentiment_result: Dict[str, Any], score: int, author_karma: int, 
                      created: datetime, subreddit: str) -> Dict[str, Any]:
        """Apply various weights to sentiment based on post quality and source."""
        base_confidence = sentiment_result.get('confidence', 0.0)
        
        # Subreddit weight
        subreddit_weight = self.subreddit_weights.get(subreddit, 0.5)
        
        # Post score weight (normalize to 0-1)
        score_weight = min(1.0, max(0.1, (score + 10) / 100))
        
        # Author karma weight
        karma_weight = min(1.0, max(0.1, author_karma / 1000))
        
        # Time decay weight (newer posts get higher weight)
        time_diff = datetime.utcnow() - created
        time_weight = max(0.1, 1.0 - (time_diff.days / 7))
        
        # Combined weight
        final_weight = subreddit_weight * score_weight * karma_weight * time_weight
        final_confidence = base_confidence * final_weight
        
        return {
            'sentiment': sentiment_result.get('sentiment', 'neutral'),
            'confidence': final_confidence,
            'weight': final_weight,
            'original_confidence': base_confidence
        }

    def _aggregate_reddit_sentiment(self, symbol: str, sentiments: List[Dict], 
                                  posts: List[Dict], total_engagement: int) -> Dict[str, Any]:
        """Aggregate sentiment scores from multiple Reddit posts."""
        if not sentiments:
            return self._create_empty_sentiment(symbol)
            
        weighted_scores = []
        sentiment_counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        subreddit_breakdown = defaultdict(lambda: {"count": 0, "sentiment": "neutral", "confidence": 0.0})
        
        for sentiment in sentiments:
            score = 0
            if sentiment['sentiment'] == 'bullish':
                score = sentiment['confidence']
                sentiment_counts["bullish"] += 1
            elif sentiment['sentiment'] == 'bearish':
                score = -sentiment['confidence']
                sentiment_counts["bearish"] += 1
            else:
                sentiment_counts["neutral"] += 1
                
            weighted_scores.append(score * sentiment['weight'])
            
        if not weighted_scores:
            return self._create_empty_sentiment(symbol)
            
        avg_score = sum(weighted_scores) / len(weighted_scores)
        avg_confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)
        
        if avg_score > 0.1:
            overall_sentiment = "bullish"
        elif avg_score < -0.1:
            overall_sentiment = "bearish"
        else:
            overall_sentiment = "neutral"
        
        # Build subreddit breakdown
        subreddit_data = defaultdict(lambda: {"posts": 0, "sentiments": [], "total_engagement": 0})
        for i, post in enumerate(posts):
            subreddit = post.get('subreddit', 'unknown')
            subreddit_data[subreddit]["posts"] += 1
            subreddit_data[subreddit]["total_engagement"] += post.get('score', 0) + post.get('comments', 0)
            if i < len(sentiments):
                subreddit_data[subreddit]["sentiments"].append(sentiments[i])
        
        # Convert subreddit data to the expected format
        subreddit_breakdown = {}
        logger.debug(f"Subreddit data: {dict(subreddit_data)}")
        for subreddit, data in subreddit_data.items():
            logger.debug(f"Processing subreddit {subreddit}: {data}")
            if data["sentiments"]:
                avg_subreddit_score = sum(s['confidence'] for s in data["sentiments"]) / len(data["sentiments"])
                subreddit_breakdown[subreddit] = {
                    "post_count": data["posts"],
                    "total_engagement": data["total_engagement"],
                    "avg_confidence": round(avg_subreddit_score, 3)
                }
            else:
                # Include subreddits even if no sentiments (posts were filtered out)
                subreddit_breakdown[subreddit] = {
                    "post_count": data["posts"],
                    "total_engagement": data["total_engagement"],
                    "avg_confidence": 0.0
                }
        
        return {
            "symbol": symbol,
            "sentiment": overall_sentiment,
            "confidence": round(avg_confidence, 3),
            "normalized_score": round(avg_score, 3),
            "post_count": len(posts),
            "sentiment_distribution": sentiment_counts,
            "total_engagement": total_engagement,
            "subreddit_breakdown": subreddit_breakdown,
            "last_updated": datetime.now().isoformat(),
            "source": "reddit"
        }

    def _create_empty_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Create empty sentiment structure."""
        return {
            "symbol": symbol,
            "sentiment": "neutral",
            "confidence": 0.0,
            "normalized_score": 0.0,
            "post_count": 0,
            "sentiment_distribution": {"bullish": 0, "bearish": 0, "neutral": 0},
            "total_engagement": 0,
            "last_updated": datetime.now().isoformat(),
            "source": "reddit"
        }

    def _cache_sentiment(self, cache_key: str, data: Dict[str, Any]):
        """Cache sentiment data in memory."""
        _sentiment_cache[cache_key] = (data, time.time())
        logger.debug(f"Cached sentiment data for key: {cache_key}")

    def _get_cached_sentiment(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get sentiment data from cache if not expired."""
        if cache_key in _sentiment_cache:
            data, timestamp = _sentiment_cache[cache_key]
            if time.time() - timestamp < _cache_expiry_seconds:
                return data
            else:
                del _sentiment_cache[cache_key]
        return None

    def clear_cache_for_symbol(self, symbol: str):
        """Clear cache for a specific symbol to force fresh fetch."""
        cache_key = f"reddit_sentiment_v3:{symbol}"
        if cache_key in _sentiment_cache:
            del _sentiment_cache[cache_key]
            logger.info(f"Cleared cache for symbol: {symbol}")
        else:
            logger.debug(f"No cache found for symbol: {symbol}")

    async def _gather_reddit_sentiment_fast(self, symbol: str) -> Dict[str, Any]:
        """Gather Reddit sentiment data for a symbol with optimized approach."""
        logger.info(f"Starting fast Reddit sentiment gathering for {symbol}")
        
        # Focus only on 24h timeframe for speed
        timeframe = "7d"
        delta = self.timeframes[timeframe]
        
        try:
            posts = await self._gather_posts_fast(symbol, delta)
            
            if not posts:
                logger.debug(f"No posts found for {symbol}")
                return self._create_empty_sentiment(symbol)

            logger.debug(f"Found {len(posts)} posts for {symbol}")
            sentiments_for_aggregation = []
            storage_tasks = []
            total_engagement = 0
            relevant_posts = []  # Track only relevant posts

            for post_data in posts:
                post_text_content = f"{post_data['title']} {post_data['text']}"
                if self._is_relevant_content(post_text_content, symbol):
                    # Analyze post sentiment using the sentiment analyzer
                    sentiment_result = await self.sentiment_analyzer.analyze_text(post_text_content)
                    
                    if "error" not in sentiment_result:
                        storage_tasks.append(self._store_raw_reddit_analysis(
                            symbol, post_text_content, "Reddit_Post", post_data.get('id', 'unknown_id'), 
                            sentiment_result, post_data['created'], post_data['subreddit']
                        ))
                        weighted_sentiment = self._apply_weights(
                            sentiment_result, post_data['score'], post_data['author_karma'],
                            post_data['created'], post_data['subreddit']
                        )
                    sentiments_for_aggregation.append(weighted_sentiment)
                    relevant_posts.append(post_data)  # Only add to relevant posts if analyzed
                    total_engagement += post_data['score'] + post_data['comments']
            
            # Wait for all raw analysis storage tasks to complete
            if storage_tasks:
                results = await asyncio.gather(*storage_tasks, return_exceptions=True)
                for i, res in enumerate(results):
                    if isinstance(res, Exception):
                        logger.error(f"Error in _store_raw_reddit_analysis task {i}: {res}")

            aggregated_result = self._aggregate_reddit_sentiment(symbol, sentiments_for_aggregation, relevant_posts, total_engagement)
            logger.info(f"Completed Reddit sentiment gathering for {symbol}. Posts: {len(relevant_posts)}, Sentiments: {len(sentiments_for_aggregation)}")
            return aggregated_result

        except Exception as e:
            logger.error(f"Error in _gather_reddit_sentiment_fast for {symbol}: {e}", exc_info=True)
            return self._create_empty_sentiment(symbol)
