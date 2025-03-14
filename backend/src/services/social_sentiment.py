import praw
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from ..config import get_settings
from .base_sentiment import BaseSentimentAnalyzer
from collections import defaultdict
import re
import asyncio
from functools import lru_cache
import time

logger = logging.getLogger(__name__)
settings = get_settings()

class SocialSentimentService(BaseSentimentAnalyzer):
    def __init__(self):
        super().__init__()
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

    async def get_reddit_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get enhanced sentiment analysis from Reddit posts and comments."""
        try:
            # Use cached data if available
            cache_key = f"{symbol}_{int(time.time() / 3600)}"  # Cache for 1 hour
            if cached := self._get_cached_sentiment(cache_key):
                return cached

            all_timeframe_data = {}
            
            # Use asyncio.wait_for to implement timeout
            for timeframe, delta in self.timeframes.items():
                try:
                    posts = await asyncio.wait_for(
                        self._gather_posts(symbol, delta),
                        timeout=self.timeout
                    )
                    if not posts:
                        all_timeframe_data[timeframe] = self._create_empty_timeframe_sentiment(symbol)
                        continue

                    post_sentiments = []
                    comment_sentiments = []
                    total_engagement = 0

                    for post in posts:
                        # Analyze post sentiment
                        post_text = f"{post['title']} {post['text']}"
                        if self._is_relevant_content(post_text, symbol):
                            sentiment = await self.analyze_text(post_text)
                            weighted_sentiment = self._apply_weights(
                                sentiment,
                                post['score'],
                                post['author_karma'],
                                post['created'],
                                post['subreddit']
                            )
                            post_sentiments.append(weighted_sentiment)
                            total_engagement += post['score'] + post['comments']

                            # Analyze top comments
                            for comment in post['top_comments']:
                                if self._is_relevant_content(comment['text'], symbol):
                                    comment_sentiment = await self.analyze_text(comment['text'])
                                    weighted_comment_sentiment = self._apply_weights(
                                        comment_sentiment,
                                        comment['score'],
                                        comment['author_karma'],
                                        comment['created'],
                                        post['subreddit']
                                    )
                                    comment_sentiments.append(weighted_comment_sentiment)
                                    total_engagement += comment['score']

                    # Combine post and comment sentiments with different weights
                    combined_sentiments = self._combine_sentiments(post_sentiments, comment_sentiments)
                    all_timeframe_data[timeframe] = self._aggregate_reddit_sentiment(
                        symbol, combined_sentiments, posts, total_engagement
                    )

                except asyncio.TimeoutError:
                    logger.error(f"Timeout while fetching Reddit data for {symbol}")
                    all_timeframe_data[timeframe] = self._create_empty_timeframe_sentiment(symbol)

            final_sentiment = self._create_final_sentiment(symbol, all_timeframe_data)
            self._cache_sentiment(cache_key, final_sentiment)
            return final_sentiment

        except Exception as e:
            logger.error(f"Reddit sentiment analysis error: {str(e)}")
            raise

    async def _gather_posts(self, symbol: str, timeframe: timedelta) -> List[Dict[str, Any]]:
        """Gather posts with rate limiting."""
        posts = []
        cutoff_time = datetime.utcnow() - timeframe

        for subreddit in self.subreddit_weights.keys():
            # Implement rate limiting
            await self._respect_rate_limit()
            
            try:
                # Use more specific search query
                query = f"title:{symbol} OR selftext:{symbol}"
                subreddit_posts = list(self.reddit.subreddit(subreddit).search(
                    query,
                    time_filter="month",  # Limit initial search
                    sort="relevance",
                    limit=10  # Reduce limit for better performance
                ))

                # Process posts
                for post in subreddit_posts:
                    if datetime.utcfromtimestamp(post.created_utc) < cutoff_time:
                        continue

                    processed_post = await self._process_post(post, subreddit)
                    if processed_post:
                        posts.append(processed_post)

            except Exception as e:
                logger.warning(f"Error fetching from r/{subreddit}: {str(e)}")
                continue

        return sorted(posts, key=lambda x: x['score'] + x['comments'], reverse=True)

    async def _process_post(self, post: Any, subreddit: str) -> Optional[Dict[str, Any]]:
        """Process a single Reddit post."""
        try:
            # Get author's karma if available
            author_karma = 0
            if post.author:
                author_karma = post.author.link_karma + post.author.comment_karma

            # Get top comments efficiently
            post.comments.replace_more(limit=0)
            top_comments = []
            for comment in list(post.comments)[:5]:  # Reduced from 10 to 5
                if comment.author:
                    comment_author_karma = comment.author.link_karma + comment.author.comment_karma
                else:
                    comment_author_karma = 0

                top_comments.append({
                    'text': comment.body,
                    'score': comment.score,
                    'author_karma': comment_author_karma,
                    'created': datetime.utcfromtimestamp(comment.created_utc)
                })

            return {
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
            logger.warning(f"Error processing post: {str(e)}")
            return None

    async def _respect_rate_limit(self):
        """Implement rate limiting for Reddit API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            await asyncio.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()

    @lru_cache(maxsize=100)
    def _get_cached_sentiment(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached sentiment data."""
        return None  # Implement proper caching if needed

    def _cache_sentiment(self, cache_key: str, sentiment_data: Dict[str, Any]):
        """Cache sentiment data."""
        pass  # Implement proper caching if needed

    def _is_relevant_content(self, text: str, symbol: str) -> bool:
        """Enhanced relevance checking with better filtering."""
        if not text or len(text) < self.min_text_length:
            return False

        # Remove URLs and special characters
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        
        # Check for spam patterns
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.spam_patterns):
            return False

        # Check symbol mention context
        symbol_lower = symbol.lower()
        words = text.split()
        if symbol_lower in words:
            idx = words.index(symbol_lower)
            context = words[max(0, idx-5):min(len(words), idx+6)]
            
            # Check for meaningful context
            context_text = ' '.join(context)
            if len(context) > 3:
                # Look for analytical terms
                analytical_terms = [
                    'analysis', 'trend', 'market', 'price', 'value',
                    'report', 'earnings', 'growth', 'revenue', 'profit',
                    'technical', 'fundamental', 'strategy', 'performance',
                    'investment', 'trading', 'volume', 'indicator'
                ]
                
                if any(term in context_text for term in analytical_terms):
                    return True

        return False

    def _apply_weights(
        self,
        sentiment: Dict[str, Any],
        score: int,
        author_karma: int,
        created_time: datetime,
        subreddit: str
    ) -> Dict[str, Any]:
        """Enhanced weight application with subreddit consideration."""
        age_hours = (datetime.utcnow() - created_time).total_seconds() / 3600
        time_decay = 1.0 / (1.0 + age_hours/24.0)  # Decay factor
        
        karma_weight = min(1.0, author_karma / 10000.0)  # Cap at 10k karma
        score_weight = min(1.0, score / 1000.0)  # Cap at 1000 score
        subreddit_weight = self.subreddit_weights.get(subreddit, 0.5)  # Default weight for unknown subreddits
        
        weighted_confidence = sentiment['confidence'] * (
            0.3 * time_decay +
            0.2 * karma_weight +
            0.2 * score_weight +
            0.3 * subreddit_weight  # Added subreddit weight
        )
        
        return {
            **sentiment,
            'weighted_confidence': weighted_confidence,
            'weight_factors': {
                'time_decay': time_decay,
                'karma_weight': karma_weight,
                'score_weight': score_weight,
                'subreddit_weight': subreddit_weight
            }
        }

    def _combine_sentiments(
        self,
        post_sentiments: List[Dict[str, Any]],
        comment_sentiments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine post and comment sentiments with weights."""
        combined = []
        
        # Weight posts higher than comments
        for sentiment in post_sentiments:
            sentiment['weighted_confidence'] *= 0.7
            combined.append(sentiment)
            
        for sentiment in comment_sentiments:
            sentiment['weighted_confidence'] *= 0.3
            combined.append(sentiment)
            
        return combined

    def _create_final_sentiment(
        self,
        symbol: str,
        timeframe_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create final sentiment analysis combining all timeframes."""
        if not any(data['post_count'] for data in timeframe_data.values()):
            return self._create_empty_timeframe_sentiment(symbol)

        # Weight recent data more heavily
        timeframe_weights = {'24h': 0.5, '7d': 0.3, '30d': 0.2}
        
        weighted_sentiment = 0
        total_weight = 0
        total_engagement = 0
        all_distributions = defaultdict(int)
        
        for timeframe, data in timeframe_data.items():
            weight = timeframe_weights[timeframe]
            if data['sentiment'] == 'bullish':
                weighted_sentiment += data['confidence'] * weight
            elif data['sentiment'] == 'bearish':
                weighted_sentiment -= data['confidence'] * weight
            total_weight += weight
            total_engagement += data['total_engagement']
            
            for sentiment, count in data['sentiment_distribution'].items():
                all_distributions[sentiment] += count

        # Determine overall sentiment
        final_sentiment = 'neutral'
        if weighted_sentiment > 0.2:
            final_sentiment = 'bullish'
        elif weighted_sentiment < -0.2:
            final_sentiment = 'bearish'

        return {
            'symbol': symbol,
            'sentiment': final_sentiment,
            'confidence': abs(weighted_sentiment / total_weight),
            'timeframes': timeframe_data,
            'total_engagement': total_engagement,
            'sentiment_distribution': dict(all_distributions),
            'last_updated': datetime.now().isoformat()
        }

    def _create_empty_timeframe_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Create empty sentiment data structure."""
        return {
            'symbol': symbol,
            'sentiment': 'neutral',
            'confidence': 0.0,
            'post_count': 0,
            'total_engagement': 0,
            'sentiment_distribution': {'bullish': 0, 'bearish': 0, 'neutral': 0},
            'timestamp': datetime.now().isoformat()
        }

    def _aggregate_reddit_sentiment(
        self,
        symbol: str,
        sentiments: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        total_engagement: int = 0
    ) -> Dict[str, Any]:
        """Aggregate sentiment analysis results from Reddit."""
        if not sentiments:
            return self._create_empty_timeframe_sentiment(symbol)
            
        sentiment_counts = {
            "bullish": sum(1 for s in sentiments if s["sentiment"] == "bullish"),
            "bearish": sum(1 for s in sentiments if s["sentiment"] == "bearish"),
            "neutral": sum(1 for s in sentiments if s["sentiment"] == "neutral")
        }

        total_confidence = sum(s.get("weighted_confidence", s["confidence"]) for s in sentiments)
        avg_confidence = total_confidence / len(sentiments)

        return {
            "symbol": symbol,
            "sentiment": max(sentiment_counts.items(), key=lambda x: x[1])[0],
            "confidence": avg_confidence,
            "source": "reddit",
            "post_count": len(posts),
            "sentiment_distribution": sentiment_counts,
            "total_engagement": total_engagement,
            "top_posts": sorted(
                [{
                    "title": p["title"],
                    "text": p["text"][:200] + "..." if len(p["text"]) > 200 else p["text"],
                    "score": p["score"],
                    "comments": p["comments"],
                    "subreddit": p["subreddit"]
                } for p in posts],
                key=lambda x: x["score"] + x["comments"],
                reverse=True
            )[:3],
            "timestamp": datetime.now().isoformat()
        }
