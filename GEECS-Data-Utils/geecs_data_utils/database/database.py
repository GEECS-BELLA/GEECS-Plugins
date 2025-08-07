"""
Define ScanDatabase class for in-memory representation of scan DB.

This module defines the ScanDatabase class, which provides an in-memory
representation of a structured collection of GEECS scan metadata.

It enables filtering, querying, and serialization of `ScanEntry` records,
allowing for high-level organization and analysis of experimental scan data.

Classes
-------
ScanDatabase
    Represents a list of ScanEntry objects with convenient access and query tools.

See Also
--------
ScanEntry : Model representing individual scan records.
ScanMetadata : Model encapsulating scan-level configuration info.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Callable
from datetime import date

from pydantic import BaseModel, Field

from geecs_data_utils.database.entries import ScanEntry


class ScanDatabase(BaseModel):
    """
    In-memory metadata database for GEECS scans.

    Attributes
    ----------
    scans : List[ScanEntry]
        List of all scan entries in the database.
    """

    scans: List[ScanEntry] = Field(default_factory=list)

    def __len__(self):
        """
        Return the number of scan entries in the database.

        Returns
        -------
        int
            The total number of scans stored.
        """
        return len(self.scans)

    def __getitem__(self, idx: int) -> ScanEntry:
        """
        Retrieve a scan entry by index.

        Parameters
        ----------
        idx : int
            Index of the desired scan.

        Returns
        -------
        ScanEntry
            The scan entry at the specified index.
        """
        return self.scans[idx]

    def __repr__(self):
        """
        Return a string representation of the database.

        Returns
        -------
        str
            Human-readable summary of the database.
        """
        return f"<ScanDatabase with {len(self.scans)} scans>"

    def summary(self, n: int = 5):
        """
        Print a brief summary of the first few scans.

        Parameters
        ----------
        n : int, optional
            Number of scans to print, by default 5.
        """
        print(f"ScanDatabase with {len(self.scans)} scans")
        for scan in self.scans[:n]:
            print(f" - {scan.scan_tag} | Devices: {scan.non_scalar_devices}")

    def to_json_file(self, path: str | Path) -> None:
        """
        Save the database to a JSON file.

        Parameters
        ----------
        path : str or Path
            Destination file path for the JSON export.
        """
        path = Path(path)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json_file(cls, path: str | Path) -> ScanDatabase:
        """
        Load the database from a JSON file.

        Parameters
        ----------
        path : str or Path
            File path of the JSON-encoded database.

        Returns
        -------
        ScanDatabase
            Loaded ScanDatabase object.
        """
        path = Path(path)
        return cls.model_validate_json(path.read_text())

    def add_entry(self, entry: ScanEntry, deduplicate: bool = True) -> None:
        """
        Append a new ScanEntry to the database, with optional deduplication.

        Parameters
        ----------
        entry : ScanEntry
            The new scan entry to add.
        deduplicate : bool, optional
            If True (default), avoid adding scans with duplicate scan_tag.
        """
        if deduplicate and any(s.scan_tag == entry.scan_tag for s in self.scans):
            return
        self.scans.append(entry)

    def query_by_device(self, device_name: str) -> List[ScanEntry]:
        """
        Retrieve scans that include a specific non-scalar device.

        Parameters
        ----------
        device_name : str
            The device name to search for.

        Returns
        -------
        List[ScanEntry]
            List of scans containing the specified device.
        """
        return [s for s in self.scans if device_name in s.non_scalar_devices]

    def query_by_scan_param(self, param: str) -> List[ScanEntry]:
        """
        Retrieve scans that used a specific scan parameter.

        Parameters
        ----------
        param : str
            The scan parameter to match.

        Returns
        -------
        List[ScanEntry]
            List of scans with the specified scan parameter.
        """
        return [s for s in self.scans if s.scan_metadata.scan_parameter == param]

    def query_by_date_range(
        self, start: date, end: date, inclusive: bool = True
    ) -> List[ScanEntry]:
        """
        Retrieve scans within a specific date range.

        Parameters
        ----------
        start : date
            Start date of the range.
        end : date
            End date of the range.
        inclusive : bool, optional
            Whether to include scans on the boundary dates. Default is True.

        Returns
        -------
        List[ScanEntry]
            List of scans matching the date filter.
        """
        if inclusive:
            return [s for s in self.scans if start <= s.date <= end]
        else:
            return [s for s in self.scans if start < s.date < end]

    def search_notes(self, keyword: str) -> List[ScanEntry]:
        """
        Search for scans whose notes contain a given keyword.

        Parameters
        ----------
        keyword : str
            Keyword to search for (case-insensitive).

        Returns
        -------
        List[ScanEntry]
            List of scans where the notes contain the keyword.
        """
        return [s for s in self.scans if s.notes and keyword.lower() in s.notes.lower()]

    def filter(self, fn: Callable[[ScanEntry], bool]) -> List[ScanEntry]:
        """
        Filter scans using a custom predicate function.

        Parameters
        ----------
        fn : Callable[[ScanEntry], bool]
            A function that takes a ScanEntry and returns True if it should be included.

        Returns
        -------
        List[ScanEntry]
            Filtered list of scan entries.
        """
        return [s for s in self.scans if fn(s)]
