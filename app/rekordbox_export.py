"""
rekordbox_export.py
====================
Generates a Rekordbox-compatible XML library file (the same format used
by iTunes-style "Import Collection" in Rekordbox: Preferences > Advanced >
Database > Import Library).

The XML contains:
  - A <COLLECTION> with one <TRACK> per file (with BPM if known)
  - A <PLAYLISTS> tree organized by Family > Subgenre, with playlists
    listing TrackIDs from the collection

The user imports this file once in Rekordbox to add tracks + playlists.
This never touches Rekordbox's own database directly (safe, no corruption risk).
"""

from __future__ import annotations

import os
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, List
from xml.dom import minidom


def _path_to_location(path: str) -> str:
    """Rekordbox expects file:// URLs with percent-encoded paths."""
    abspath = os.path.abspath(path)
    # Rekordbox XML uses forward slashes even on Windows
    abspath = abspath.replace("\\", "/")
    if not abspath.startswith("/"):
        abspath = "/" + abspath
    quoted = urllib.parse.quote(abspath)
    return f"file://localhost{quoted}"


def build_rekordbox_xml(
    tracks: List[Dict],
    out_path: str,
    grouping: str = "family_then_subgenre",
):
    """
    tracks: list of dicts with keys:
        path, subgenre, family, bpm (optional), duration_sec (optional),
        title (optional), artist (optional)

    grouping:
        "family_then_subgenre" -> nested playlist folders: Family / Subgenre
        "subgenre"             -> flat playlists, one per subgenre
        "family"               -> flat playlists, one per family
    """
    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")

    ET.SubElement(root, "PRODUCT", Name="MilkCrate", Version="1.0", Company="MilkCrate")

    collection = ET.SubElement(root, "COLLECTION", Entries=str(len(tracks)))

    track_id_map: Dict[int, Dict] = {}
    for idx, t in enumerate(tracks, start=1):
        track_id_map[idx] = t
        attrs = {
            "TrackID": str(idx),
            "Name": t.get("title") or os.path.splitext(os.path.basename(t["path"]))[0],
            "Artist": t.get("artist", ""),
            "Location": _path_to_location(t["path"]),
            "Genre": t.get("subgenre", ""),
        }
        if t.get("bpm"):
            attrs["AverageBpm"] = f"{t['bpm']:.2f}"
        if t.get("duration_sec"):
            attrs["TotalTime"] = str(int(t["duration_sec"]))
        ET.SubElement(collection, "TRACK", **attrs)

    playlists_root = ET.SubElement(root, "PLAYLISTS")
    root_node = ET.SubElement(playlists_root, "NODE", Type="0", Name="ROOT", Count="0")

    def make_playlist_node(parent, name, track_ids):
        node = ET.SubElement(parent, "NODE", Name=name, Type="1", KeyType="0",
                              Entries=str(len(track_ids)))
        for tid in track_ids:
            ET.SubElement(node, "TRACK", Key=str(tid))
        return node

    if grouping == "family_then_subgenre":
        # group track ids by family then subgenre
        families: Dict[str, Dict[str, List[int]]] = {}
        for idx, t in track_id_map.items():
            fam = t.get("family", "Unknown")
            sub = t.get("subgenre", "Unknown")
            families.setdefault(fam, {}).setdefault(sub, []).append(idx)

        folder_count = 0
        for fam, subs in families.items():
            folder = ET.SubElement(root_node, "NODE", Type="0", Name=fam,
                                    Count=str(len(subs)))
            folder_count += 1
            for sub, ids in subs.items():
                make_playlist_node(folder, sub, ids)
        root_node.set("Count", str(folder_count))

    elif grouping == "subgenre":
        groups: Dict[str, List[int]] = {}
        for idx, t in track_id_map.items():
            groups.setdefault(t.get("subgenre", "Unknown"), []).append(idx)
        for name, ids in groups.items():
            make_playlist_node(root_node, name, ids)
        root_node.set("Count", str(len(groups)))

    elif grouping == "family":
        groups: Dict[str, List[int]] = {}
        for idx, t in track_id_map.items():
            groups.setdefault(t.get("family", "Unknown"), []).append(idx)
        for name, ids in groups.items():
            make_playlist_node(root_node, name, ids)
        root_node.set("Count", str(len(groups)))

    else:
        raise ValueError(f"Unknown grouping: {grouping}")

    # Pretty-print
    rough = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")

    with open(out_path, "wb") as f:
        f.write(pretty)

    return out_path
