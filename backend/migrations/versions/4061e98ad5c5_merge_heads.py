"""Merge heads

Revision ID: 4061e98ad5c5
Revises: 74683fe48775, add_aggregated_sentiment_table, d3596633537c
Create Date: 2025-06-17 13:16:00.573455

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4061e98ad5c5'
down_revision: Union[str, None] = ('74683fe48775', 'add_aggregated_sentiment_table', 'd3596633537c')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
