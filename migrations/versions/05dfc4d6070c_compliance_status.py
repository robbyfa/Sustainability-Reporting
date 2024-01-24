"""Compliance Status

Revision ID: 05dfc4d6070c
Revises: 867c4ad56af6
Create Date: 2024-01-23 15:05:53.643137

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '05dfc4d6070c'
down_revision = '867c4ad56af6'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('activity_criteria', schema=None) as batch_op:
        batch_op.add_column(sa.Column('compliance_status', sa.String(length=50), nullable=True))
        batch_op.drop_column('is_compliant')

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('activity_criteria', schema=None) as batch_op:
        batch_op.add_column(sa.Column('is_compliant', sa.BOOLEAN(), nullable=True))
        batch_op.drop_column('compliance_status')

    # ### end Alembic commands ###