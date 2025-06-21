"""
Alembic migration script to add the aggregated_sentiments table for MacroMind.

Revision ID: add_aggregated_sentiment_table
Revises: 
Create Date: 2025-06-17
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_aggregated_sentiment_table'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'aggregated_sentiments',
        sa.Column('id', sa.Integer(), primary_key=True, index=True),
        sa.Column('symbol', sa.String(), index=True),
        sa.Column('date', sa.Date(), index=True),
        sa.Column('sentiment', sa.Enum('bullish', 'bearish', 'neutral', name='sentimenttype', create_type=False, native_enum=False), nullable=True),
        sa.Column('score', sa.Float(), nullable=True),
        sa.Column('avg_daily_score', sa.Float(), nullable=True),
        sa.Column('moving_avg_7d', sa.Float(), nullable=True),
        sa.Column('news_score', sa.Float(), nullable=True),
        sa.Column('reddit_score', sa.Float(), nullable=True),
        sa.Column('benchmark', sa.String(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
    )
    # op.create_index('ix_aggregated_sentiments_symbol', 'aggregated_sentiments', ['symbol'])
    # op.create_index('ix_aggregated_sentiments_date', 'aggregated_sentiments', ['date'])

def downgrade():
    # op.drop_index('ix_aggregated_sentiments_symbol', table_name='aggregated_sentiments')
    # op.drop_index('ix_aggregated_sentiments_date', table_name='aggregated_sentiments')
    op.drop_table('aggregated_sentiments')
