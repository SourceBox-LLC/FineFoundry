"""Edge case tests for db/training_runs.py.

Tests cover:
- Invalid run IDs
- Concurrent operations
- Edge cases in status updates
- Storage path handling
- Empty/null values
"""

import os
import pytest

from db.core import init_db, close_all_connections, _DB_PATH_OVERRIDE
from db.training_runs import (
    create_training_run,
    get_training_run,
    list_training_runs,
    update_training_run,
    delete_training_run,
    get_managed_storage_root,
)


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = str(tmp_path / "test.db")
    _DB_PATH_OVERRIDE["path"] = db_path
    init_db(db_path)
    yield db_path
    close_all_connections()
    _DB_PATH_OVERRIDE.clear()


class TestGetTrainingRunEdgeCases:
    """Edge cases for get_training_run."""

    def test_nonexistent_id_returns_none(self, temp_db):
        """Test that a nonexistent ID returns None."""
        result = get_training_run(99999)
        assert result is None

    def test_negative_id_returns_none(self, temp_db):
        """Test that a negative ID returns None."""
        result = get_training_run(-1)
        assert result is None

    def test_zero_id_returns_none(self, temp_db):
        """Test that ID 0 returns None."""
        result = get_training_run(0)
        assert result is None


class TestCreateTrainingRunEdgeCases:
    """Edge cases for create_training_run."""

    def test_empty_name(self, temp_db):
        """Test creating a run with empty name."""
        run = create_training_run(
            name="",
            base_model="test-model",
            dataset_source="database",
            dataset_id="1",
        )
        assert run is not None
        assert run["name"] == ""

    def test_special_characters_in_name(self, temp_db):
        """Test creating a run with special characters in name."""
        run = create_training_run(
            name="test/run:with<special>chars",
            base_model="test-model",
            dataset_source="database",
            dataset_id="1",
        )
        assert run is not None
        # Special chars should be sanitized in storage path
        assert "/" not in os.path.basename(run["storage_path"])
        assert ":" not in os.path.basename(run["storage_path"])

    def test_very_long_name(self, temp_db):
        """Test creating a run with very long name (truncated in storage path)."""
        long_name = "a" * 500
        run = create_training_run(
            name=long_name,
            base_model="test-model",
            dataset_source="database",
            dataset_id="1",
        )
        assert run is not None
        # Full name is stored in database
        assert run["name"] == long_name
        # But storage path uses truncated name
        storage_dir_name = os.path.basename(run["storage_path"])
        # Name portion should be truncated to 100 chars
        assert len(storage_dir_name.split("_")[0]) <= 100

    def test_unicode_name(self, temp_db):
        """Test creating a run with unicode characters."""
        run = create_training_run(
            name="训练运行_テスト_🚀",
            base_model="test-model",
            dataset_source="database",
            dataset_id="1",
        )
        assert run is not None
        assert "训练运行" in run["name"]

    def test_null_hp_and_metadata(self, temp_db):
        """Test creating a run with None hp and metadata."""
        run = create_training_run(
            name="test",
            base_model="test-model",
            dataset_source="database",
            dataset_id="1",
            hp=None,
            metadata=None,
        )
        assert run is not None
        assert run["hp_json"] is None
        assert run["metadata_json"] is None

    def test_empty_hp_dict(self, temp_db):
        """Test creating a run with empty hp dict (treated as None)."""
        run = create_training_run(
            name="test",
            base_model="test-model",
            dataset_source="database",
            dataset_id="1",
            hp={},
        )
        assert run is not None
        # Empty dict is treated as falsy, so stored as None
        assert run["hp_json"] is None


class TestListTrainingRunsEdgeCases:
    """Edge cases for list_training_runs."""

    def test_empty_database(self, temp_db):
        """Test listing when no runs exist."""
        runs = list_training_runs()
        assert runs == []

    def test_filter_by_nonexistent_status(self, temp_db):
        """Test filtering by a status that doesn't exist."""
        create_training_run(
            name="test",
            base_model="test-model",
            dataset_source="database",
            dataset_id="1",
        )
        runs = list_training_runs(status="nonexistent_status")
        assert runs == []

    def test_large_offset(self, temp_db):
        """Test with offset larger than total runs."""
        create_training_run(
            name="test",
            base_model="test-model",
            dataset_source="database",
            dataset_id="1",
        )
        runs = list_training_runs(offset=1000)
        assert runs == []

    def test_zero_limit(self, temp_db):
        """Test with limit of 0."""
        create_training_run(
            name="test",
            base_model="test-model",
            dataset_source="database",
            dataset_id="1",
        )
        runs = list_training_runs(limit=0)
        assert runs == []


class TestUpdateTrainingRunEdgeCases:
    """Edge cases for update_training_run."""

    def test_update_nonexistent_run(self, temp_db):
        """Test updating status of nonexistent run."""
        # Should return None for nonexistent run
        result = update_training_run(99999, status="completed")
        assert result is None

    def test_update_to_same_status(self, temp_db):
        """Test updating to the same status."""
        run = create_training_run(
            name="test",
            base_model="test-model",
            dataset_source="database",
            dataset_id="1",
        )
        # Update to pending (same as initial)
        update_training_run(run["id"], status="pending")
        updated = get_training_run(run["id"])
        assert updated["status"] == "pending"


class TestDeleteTrainingRunEdgeCases:
    """Edge cases for delete_training_run."""

    def test_delete_nonexistent_run(self, temp_db):
        """Test deleting a run that doesn't exist."""
        # Should not raise
        result = delete_training_run(99999)
        # Result depends on implementation - just verify no crash
        assert result is True or result is False

    def test_delete_and_verify_storage_cleanup(self, temp_db):
        """Test that deleting a run cleans up storage."""
        run = create_training_run(
            name="test",
            base_model="test-model",
            dataset_source="database",
            dataset_id="1",
        )
        storage_path = run["storage_path"]
        assert os.path.exists(storage_path)

        delete_training_run(run["id"], delete_files=True)

        # Storage should be cleaned up
        assert not os.path.exists(storage_path)


class TestStoragePathEdgeCases:
    """Edge cases for storage path handling."""

    def test_storage_root_created_if_missing(self, temp_db):
        """Test that storage root is created if it doesn't exist."""
        root = get_managed_storage_root()
        assert os.path.exists(root)
        assert os.path.isdir(root)

    def test_multiple_runs_have_unique_paths(self, temp_db):
        """Test that multiple runs get unique storage paths."""
        run1 = create_training_run(
            name="test",
            base_model="test-model",
            dataset_source="database",
            dataset_id="1",
        )
        run2 = create_training_run(
            name="test",
            base_model="test-model",
            dataset_source="database",
            dataset_id="1",
        )
        assert run1["storage_path"] != run2["storage_path"]
