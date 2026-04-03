"""Google Drive Document ID lookup using experiment-specific .tsv files.

This module provides date-aware Document ID resolution by downloading and caching
.tsv files from Google Drive that map dates (MM-DD-YY format) to Document IDs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Mapping from experiment name to Google Drive file ID for doc_index.tsv
EXPERIMENT_FILE_IDS: Dict[str, str] = {
    "Undulator": "1sePxxsUfsX3_gedS9xab9Crdt4fVxWMh",
    # "Thomson": "...",  # Add as needed
}


class DocIDLookup:
    """Download and cache Google Drive doc_index.tsv files for date-aware Document ID lookup.

    Each experiment has its own .tsv file on Google Drive that maps dates (MM-DD-YY format)
    to Document IDs. This class handles downloading, caching, and querying these files.

    Parameters
    ----------
    experiment : str
        Name of the experiment (e.g., "Undulator", "Thomson").
    file_id : str
        Google Drive file ID for the experiment's doc_index.tsv file.
    cache_dir : Path, optional
        Directory to cache downloaded .tsv files. Defaults to ~/.cache/LiveWatchGUI/
    """

    def __init__(
        self,
        experiment: str,
        file_id: str,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize DocIDLookup with experiment name and Google Drive file ID."""
        self.experiment = experiment
        self.file_id = file_id

        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "LiveWatchGUI"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache file path
        self.cache_file = self.cache_dir / f"{experiment}_doc_index.tsv"

        # Date-to-DocID mapping (loaded on demand)
        self._mapping: Dict[str, str] = {}
        self._loaded = False

    def get_document_id(self, year: int, month: int, day: int) -> Optional[str]:
        """Look up Document ID for a specific date.

        Parameters
        ----------
        year : int
            Year (e.g., 2026).
        month : int
            Month (1-12).
        day : int
            Day (1-31).

        Returns
        -------
        Optional[str]
            Document ID if found, None otherwise.
        """
        # Load mapping if not already loaded
        if not self._loaded:
            self._load_mapping()

        # Format date as MM-DD-YY
        date_key = f"{month:02d}-{day:02d}-{year % 100:02d}"

        return self._mapping.get(date_key)

    def refresh(self, force_download: bool = True) -> bool:
        """Download latest .tsv file from Google Drive and update mapping.

        Parameters
        ----------
        force_download : bool, optional
            If True (default), always download from Google Drive.
            If False, use cache if available.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        logger.info("Refreshing doc_index.tsv for experiment '%s'...", self.experiment)

        # Try to download from Google Drive (unless force_download is False and cache exists)
        if force_download or not self.cache_file.exists():
            if not self._download_from_drive():
                logger.warning(
                    "Failed to download doc_index.tsv for experiment '%s'",
                    self.experiment,
                )
                # Fall back to cache if download fails
                if not self.cache_file.exists():
                    return False

        if not self._load_from_cache():
            logger.warning(
                "Failed to load cached doc_index.tsv for experiment '%s'",
                self.experiment,
            )
            return False

        logger.info(
            "Successfully loaded doc_index.tsv for experiment '%s' with %d entries",
            self.experiment,
            len(self._mapping),
        )
        return True

    def _load_mapping(self) -> None:
        """Load mapping from cache or download if not cached."""
        if self._load_from_cache():
            self._loaded = True
            return

        # Try to download if cache doesn't exist
        if self._download_from_drive():
            self._load_from_cache()
            self._loaded = True
        else:
            logger.warning(
                "Could not load or download doc_index.tsv for experiment '%s'",
                self.experiment,
            )
            self._loaded = True  # Mark as loaded to avoid repeated attempts

    def _load_from_cache(self) -> bool:
        """Load mapping from cached .tsv file if it exists.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if not self.cache_file.exists():
            logger.debug("Cache file does not exist: %s", self.cache_file)
            return False

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                content = f.read()
            self._mapping = self._parse_tsv(content)
            logger.debug(
                "Loaded %d entries from cache file: %s",
                len(self._mapping),
                self.cache_file,
            )
            return True
        except Exception as exc:
            logger.warning("Error loading cache file %s: %s", self.cache_file, exc)
            return False

    def _download_from_drive(self) -> bool:
        """Download .tsv file from Google Drive.

        Uses the Google Drive export URL to download the file as TSV format.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        try:
            import requests
        except ImportError:
            logger.error(
                "requests library not available; cannot download from Google Drive"
            )
            return False

        # Google Drive direct download URL (more reliable than export URL)
        url = f"https://drive.google.com/uc?export=download&id={self.file_id}"

        try:
            logger.debug("Downloading doc_index.tsv from Google Drive: %s", url)

            # Add User-Agent header to avoid being blocked by Google Drive
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            response = requests.get(
                url, timeout=10, headers=headers, allow_redirects=True
            )
            response.raise_for_status()

            # Save to cache
            with open(self.cache_file, "w", encoding="utf-8") as f:
                f.write(response.text)

            logger.debug("Successfully downloaded and cached: %s", self.cache_file)
            return True
        except Exception as exc:
            logger.warning(
                "Error downloading from Google Drive (file_id=%s): %s",
                self.file_id,
                exc,
            )
            return False

    def _parse_tsv(self, content: str) -> Dict[str, str]:
        """Parse TSV content and return date-to-DocID mapping.

        Expected format:
            Date (MM-DD-YY) | Document ID
            01-15-26        | 1abc2def3ghi4jkl5mno6pqr7stu8vwx
            01-16-26        | 2bcd3efg4hij5klm6nop7qrs8tuv9wxy
            ...

        Parameters
        ----------
        content : str
            TSV file content.

        Returns
        -------
        Dict[str, str]
            Mapping from date strings (MM-DD-YY) to Document IDs.
        """
        mapping = {}
        lines = content.strip().split("\n")

        for i, line in enumerate(lines):
            # Skip header row (first line)
            if i == 0:
                continue

            # Skip empty lines
            if not line.strip():
                continue

            # Parse TSV line (tab-separated)
            parts = line.split("\t")
            if len(parts) < 2:
                logger.debug("Skipping malformed TSV line %d: %s", i + 1, line)
                continue

            date_str = parts[0].strip()
            doc_id = parts[1].strip()

            if date_str and doc_id:
                mapping[date_str] = doc_id

        logger.debug("Parsed %d date-to-DocID mappings from TSV", len(mapping))
        return mapping
