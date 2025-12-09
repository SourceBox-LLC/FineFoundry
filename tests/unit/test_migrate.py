"""Unit tests for db/migrate.py.

Tests cover:
- migrate_from_json() - full migration
- is_migration_complete() - migration status check
- _migrate_settings() - settings migration
- _migrate_training_configs() - config migration
- _migrate_scraped_data() - scraped data migration
- export_all_to_json() - JSON export
"""

import json
import os

import pytest

from db.core import init_db, close_all_connections, _DB_PATH_OVERRIDE
from db.migrate import (
    migrate_from_json,
    is_migration_complete,
    export_all_to_json,
    _migrate_settings,
    _migrate_training_configs,
    _migrate_scraped_data,
)
from db.settings import get_setting
from db.scraped_data import get_all_pairs


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = str(tmp_path / "test.db")
    _DB_PATH_OVERRIDE["path"] = db_path
    init_db(db_path)
    yield db_path
    close_all_connections()
    _DB_PATH_OVERRIDE.clear()


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project directory with JSON files."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Create ff_settings.json
    settings = {
        "huggingface": {"token": "hf_test_token"},
        "runpod": {"api_key": "rp_test_key"},
        "simple_key": "simple_value",
    }
    with open(project_dir / "ff_settings.json", "w") as f:
        json.dump(settings, f)

    # Create saved_configs directory
    configs_dir = project_dir / "saved_configs"
    configs_dir.mkdir()

    config1 = {"model": "llama", "epochs": 3}
    with open(configs_dir / "config1.json", "w") as f:
        json.dump(config1, f)

    config2 = {"model": "mistral", "epochs": 5}
    with open(configs_dir / "config2.json", "w") as f:
        json.dump(config2, f)

    # Create scraped data file
    scraped_data = [
        {"input": "Question 1", "output": "Answer 1"},
        {"input": "Question 2", "output": "Answer 2"},
    ]
    with open(project_dir / "scraped_training_data.json", "w") as f:
        json.dump(scraped_data, f)

    return str(project_dir)


# =============================================================================
# migrate_from_json() tests
# =============================================================================


class TestMigrateFromJson:
    """Tests for full migration."""

    def test_full_migration(self, temp_db, temp_project):
        """Test full migration from JSON to SQLite."""
        results = migrate_from_json(project_root=temp_project, db_path=temp_db)

        assert results["settings"]["migrated"] is True
        assert results["training_configs"]["migrated"] == 2
        assert results["scraped_data"]["migrated"] == 2

    def test_marks_migration_complete(self, temp_db, temp_project):
        """Test that migration is marked complete."""
        migrate_from_json(project_root=temp_project, db_path=temp_db)

        assert is_migration_complete(db_path=temp_db) is True

    def test_migration_with_missing_files(self, temp_db, tmp_path):
        """Test migration with missing JSON files."""
        empty_project = tmp_path / "empty"
        empty_project.mkdir()

        results = migrate_from_json(project_root=str(empty_project), db_path=temp_db)

        # Should not crash, just report errors
        assert results["settings"]["error"] is not None
        assert "not found" in results["settings"]["error"]


# =============================================================================
# is_migration_complete() tests
# =============================================================================


class TestIsMigrationComplete:
    """Tests for migration status check."""

    def test_not_complete_initially(self, temp_db):
        """Test that migration is not complete initially."""
        assert is_migration_complete(db_path=temp_db) is False

    def test_complete_after_migration(self, temp_db, temp_project):
        """Test that migration is complete after running."""
        migrate_from_json(project_root=temp_project, db_path=temp_db)
        assert is_migration_complete(db_path=temp_db) is True


# =============================================================================
# _migrate_settings() tests
# =============================================================================


class TestMigrateSettings:
    """Tests for settings migration."""

    def test_migrate_nested_settings(self, temp_db, temp_project):
        """Test migrating nested settings."""
        result = _migrate_settings(temp_project, temp_db)

        assert result["migrated"] is True
        assert get_setting("huggingface.token", db_path=temp_db) == "hf_test_token"
        assert get_setting("runpod.api_key", db_path=temp_db) == "rp_test_key"

    def test_migrate_simple_settings(self, temp_db, temp_project):
        """Test migrating simple (non-nested) settings."""
        result = _migrate_settings(temp_project, temp_db)

        assert result["migrated"] is True
        assert get_setting("simple_key", db_path=temp_db) == "simple_value"

    def test_missing_settings_file(self, temp_db, tmp_path):
        """Test with missing settings file."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = _migrate_settings(str(empty_dir), temp_db)

        assert result["migrated"] is False
        assert "not found" in result["error"]

    def test_invalid_settings_format(self, temp_db, tmp_path):
        """Test with invalid settings format."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create invalid settings (list instead of dict)
        with open(project_dir / "ff_settings.json", "w") as f:
            json.dump(["invalid", "format"], f)

        result = _migrate_settings(str(project_dir), temp_db)

        assert result["migrated"] is False
        assert "Invalid" in result["error"]


# =============================================================================
# _migrate_training_configs() tests
# =============================================================================


class TestMigrateTrainingConfigs:
    """Tests for training config migration."""

    def test_migrate_configs(self, temp_db, temp_project):
        """Test migrating training configs."""
        result = _migrate_training_configs(temp_project, temp_db)

        assert result["migrated"] == 2
        assert len(result["errors"]) == 0

    def test_missing_configs_dir(self, temp_db, tmp_path):
        """Test with missing configs directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = _migrate_training_configs(str(empty_dir), temp_db)

        assert result["migrated"] == 0
        assert len(result["errors"]) > 0

    def test_invalid_config_file(self, temp_db, tmp_path):
        """Test with invalid config file."""
        project_dir = tmp_path / "project"
        configs_dir = project_dir / "saved_configs"
        configs_dir.mkdir(parents=True)

        # Create invalid JSON
        with open(configs_dir / "invalid.json", "w") as f:
            f.write("not valid json")

        result = _migrate_training_configs(str(project_dir), temp_db)

        assert len(result["errors"]) > 0


# =============================================================================
# _migrate_scraped_data() tests
# =============================================================================


class TestMigrateScrapedData:
    """Tests for scraped data migration."""

    def test_migrate_standard_format(self, temp_db, temp_project):
        """Test migrating standard format pairs."""
        result = _migrate_scraped_data(temp_project, temp_db)

        assert result["migrated"] == 2
        assert result["sessions"] == 1

    def test_migrate_chatml_format(self, temp_db, tmp_path):
        """Test migrating ChatML format data."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create ChatML format data
        chatml_data = [
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        ]
        with open(project_dir / "scraped_training_data.json", "w") as f:
            json.dump(chatml_data, f)

        result = _migrate_scraped_data(str(project_dir), temp_db)

        assert result["migrated"] == 1

    def test_source_detection_reddit(self, temp_db, tmp_path):
        """Test source detection for Reddit files."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with open(project_dir / "reddit_pairs.json", "w") as f:
            json.dump([{"input": "Q", "output": "A"}], f)

        result = _migrate_scraped_data(str(project_dir), temp_db)

        assert result["sessions"] == 1

    def test_source_detection_fourchan(self, temp_db, tmp_path):
        """Test source detection for 4chan files."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with open(project_dir / "fourchan_pairs.json", "w") as f:
            json.dump([{"input": "Q", "output": "A"}], f)

        result = _migrate_scraped_data(str(project_dir), temp_db)

        assert result["sessions"] == 1

    def test_no_scraped_files(self, temp_db, tmp_path):
        """Test with no scraped data files."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = _migrate_scraped_data(str(empty_dir), temp_db)

        assert result["migrated"] == 0
        assert result["sessions"] == 0


# =============================================================================
# export_all_to_json() tests
# =============================================================================


class TestExportAllToJson:
    """Tests for JSON export."""

    def test_export_after_migration(self, temp_db, temp_project, tmp_path):
        """Test exporting data after migration."""
        # First migrate
        migrate_from_json(project_root=temp_project, db_path=temp_db)

        # Then export
        output_dir = str(tmp_path / "export")
        outputs = export_all_to_json(output_dir, db_path=temp_db)

        assert "settings" in outputs
        assert os.path.exists(outputs["settings"])

        # Verify settings content
        with open(outputs["settings"]) as f:
            exported = json.load(f)
        assert "huggingface" in exported

    def test_export_scraped_data(self, temp_db, temp_project, tmp_path):
        """Test exporting scraped data."""
        migrate_from_json(project_root=temp_project, db_path=temp_db)

        output_dir = str(tmp_path / "export")
        outputs = export_all_to_json(output_dir, db_path=temp_db)

        assert "scraped_data" in outputs
        assert os.path.exists(outputs["scraped_data"])

        with open(outputs["scraped_data"]) as f:
            exported = json.load(f)
        assert len(exported) == 2

    def test_export_creates_directory(self, temp_db, tmp_path):
        """Test that export creates output directory."""
        output_dir = str(tmp_path / "new" / "nested" / "dir")
        export_all_to_json(output_dir, db_path=temp_db)

        assert os.path.exists(output_dir)


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_unicode_in_settings(self, temp_db, tmp_path):
        """Test unicode content in settings."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        settings = {"key": "日本語の値"}
        with open(project_dir / "ff_settings.json", "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False)

        result = _migrate_settings(str(project_dir), temp_db)

        assert result["migrated"] is True
        assert get_setting("key", db_path=temp_db) == "日本語の値"

    def test_unicode_in_scraped_data(self, temp_db, tmp_path):
        """Test unicode content in scraped data."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        data = [{"input": "日本語の質問", "output": "日本語の回答"}]
        with open(project_dir / "scraped_training_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        result = _migrate_scraped_data(str(project_dir), temp_db)

        assert result["migrated"] == 1
        pairs = get_all_pairs(db_path=temp_db)
        assert pairs[0]["input"] == "日本語の質問"

    def test_empty_pairs_skipped(self, temp_db, tmp_path):
        """Test that empty pairs are skipped."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        data = [
            {"input": "", "output": "Answer"},
            {"input": "Question", "output": ""},
            {"input": "Valid", "output": "Pair"},
        ]
        with open(project_dir / "scraped_training_data.json", "w") as f:
            json.dump(data, f)

        result = _migrate_scraped_data(str(project_dir), temp_db)

        assert result["migrated"] == 1

    def test_migration_idempotent(self, temp_db, temp_project):
        """Test that running migration twice doesn't duplicate data."""
        # Run migration twice
        migrate_from_json(project_root=temp_project, db_path=temp_db)
        migrate_from_json(project_root=temp_project, db_path=temp_db)

        # Should still be marked complete
        assert is_migration_complete(db_path=temp_db) is True
