"""initial schema

Revision ID: d3596633537c
Revises: 
Create Date: 2025-05-13 11:21:43.795665

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd3596633537c'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('economic_events',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=True),
    sa.Column('date', sa.DateTime(), nullable=True),
    sa.Column('description', sa.String(), nullable=True),
    sa.Column('impact', sa.String(), nullable=True),
    sa.Column('forecast', sa.Float(), nullable=True),
    sa.Column('previous', sa.Float(), nullable=True),
    sa.Column('actual', sa.Float(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_date_impact', 'economic_events', ['date', 'impact'], unique=False)
    op.create_index(op.f('ix_economic_events_id'), 'economic_events', ['id'], unique=False)
    op.create_table('market_sentiments',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('symbol', sa.String(), nullable=True),
    sa.Column('date', sa.Date(), nullable=True),
    sa.Column('sentiment', sa.Enum('BULLISH', 'BEARISH', 'NEUTRAL', name='sentimenttype'), nullable=True),
    sa.Column('score', sa.Float(), nullable=True),
    sa.Column('avg_daily_score', sa.Float(), nullable=True),
    sa.Column('moving_avg_7d', sa.Float(), nullable=True),
    sa.Column('news_score', sa.Float(), nullable=True),
    sa.Column('reddit_score', sa.Float(), nullable=True),
    sa.Column('benchmark', sa.String(), nullable=True),
    sa.Column('market_condition', sa.String(), nullable=True),
    sa.Column('volatility_level', sa.Float(), nullable=True),
    sa.Column('timestamp', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_symbol_date', 'market_sentiments', ['symbol', 'date'], unique=False)
    op.create_index(op.f('ix_market_sentiments_date'), 'market_sentiments', ['date'], unique=False)
    op.create_index(op.f('ix_market_sentiments_id'), 'market_sentiments', ['id'], unique=False)
    op.create_index(op.f('ix_market_sentiments_symbol'), 'market_sentiments', ['symbol'], unique=False)
    op.create_table('raw_sentiment_analyses',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('symbol', sa.String(), nullable=True),
    sa.Column('text_content_hash', sa.String(), nullable=True),
    sa.Column('text_content', sa.String(), nullable=True),
    sa.Column('source', sa.String(), nullable=True),
    sa.Column('sentiment_label', sa.String(), nullable=True),
    sa.Column('sentiment_score', sa.Float(), nullable=True),
    sa.Column('all_scores', sa.JSON(), nullable=True),
    sa.Column('analyzed_at', sa.DateTime(), nullable=True),
    sa.Column('source_created_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_raw_sentiment_symbol_source_created', 'raw_sentiment_analyses', ['symbol', 'source', 'source_created_at'], unique=False)
    op.create_index(op.f('ix_raw_sentiment_analyses_analyzed_at'), 'raw_sentiment_analyses', ['analyzed_at'], unique=False)
    op.create_index(op.f('ix_raw_sentiment_analyses_id'), 'raw_sentiment_analyses', ['id'], unique=False)
    op.create_index(op.f('ix_raw_sentiment_analyses_source'), 'raw_sentiment_analyses', ['source'], unique=False)
    op.create_index(op.f('ix_raw_sentiment_analyses_source_created_at'), 'raw_sentiment_analyses', ['source_created_at'], unique=False)
    op.create_index(op.f('ix_raw_sentiment_analyses_symbol'), 'raw_sentiment_analyses', ['symbol'], unique=False)
    op.create_index(op.f('ix_raw_sentiment_analyses_text_content_hash'), 'raw_sentiment_analyses', ['text_content_hash'], unique=True)
    op.create_table('sector_fundamentals',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('sector_name', sa.String(), nullable=True),
    sa.Column('date', sa.Date(), nullable=True),
    sa.Column('pe_ratio', sa.Float(), nullable=True),
    sa.Column('pb_ratio', sa.Float(), nullable=True),
    sa.Column('earnings_growth', sa.Float(), nullable=True),
    sa.Column('timestamp', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_sector_date', 'sector_fundamentals', ['sector_name', 'date'], unique=False)
    op.create_index(op.f('ix_sector_fundamentals_date'), 'sector_fundamentals', ['date'], unique=False)
    op.create_index(op.f('ix_sector_fundamentals_id'), 'sector_fundamentals', ['id'], unique=False)
    op.create_index(op.f('ix_sector_fundamentals_sector_name'), 'sector_fundamentals', ['sector_name'], unique=False)
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('email', sa.String(), nullable=True),
    sa.Column('hashed_password', sa.String(), nullable=True),
    sa.Column('is_vip', sa.Boolean(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_table('alerts',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('name', sa.String(), nullable=True),
    sa.Column('symbol', sa.String(), nullable=True),
    sa.Column('conditions', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('last_triggered_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_alerts_id'), 'alerts', ['id'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_alerts_id'), table_name='alerts')
    op.drop_table('alerts')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
    op.drop_index(op.f('ix_sector_fundamentals_sector_name'), table_name='sector_fundamentals')
    op.drop_index(op.f('ix_sector_fundamentals_id'), table_name='sector_fundamentals')
    op.drop_index(op.f('ix_sector_fundamentals_date'), table_name='sector_fundamentals')
    op.drop_index('idx_sector_date', table_name='sector_fundamentals')
    op.drop_table('sector_fundamentals')
    op.drop_index(op.f('ix_raw_sentiment_analyses_text_content_hash'), table_name='raw_sentiment_analyses')
    op.drop_index(op.f('ix_raw_sentiment_analyses_symbol'), table_name='raw_sentiment_analyses')
    op.drop_index(op.f('ix_raw_sentiment_analyses_source_created_at'), table_name='raw_sentiment_analyses')
    op.drop_index(op.f('ix_raw_sentiment_analyses_source'), table_name='raw_sentiment_analyses')
    op.drop_index(op.f('ix_raw_sentiment_analyses_id'), table_name='raw_sentiment_analyses')
    op.drop_index(op.f('ix_raw_sentiment_analyses_analyzed_at'), table_name='raw_sentiment_analyses')
    op.drop_index('idx_raw_sentiment_symbol_source_created', table_name='raw_sentiment_analyses')
    op.drop_table('raw_sentiment_analyses')
    op.drop_index(op.f('ix_market_sentiments_symbol'), table_name='market_sentiments')
    op.drop_index(op.f('ix_market_sentiments_id'), table_name='market_sentiments')
    op.drop_index(op.f('ix_market_sentiments_date'), table_name='market_sentiments')
    op.drop_index('idx_symbol_date', table_name='market_sentiments')
    op.drop_table('market_sentiments')
    op.drop_index(op.f('ix_economic_events_id'), table_name='economic_events')
    op.drop_index('idx_date_impact', table_name='economic_events')
    op.drop_table('economic_events')
    # ### end Alembic commands ###
