"""Export a professional, fully local recorded-mission replay workspace."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

from orbexa.simulation.mission_result import ADTMPCMissionResult
from orbexa.visualization.studio_data import StudioDataset


@dataclass
class StudioEntry:
    """One saved result and optional existing media within the export root."""

    key: str
    title: str
    result: Any
    data_href: str
    video_href: str | None = None
    evidence_href: str | None = None
    lifecycle_href: str | None = None
    figures: Sequence[Mapping[str, str]] = ()


class MissionStudio:
    """Static exporter; web assets are distributed with the Python package.

    Link paths are relative to ``output_dir/studio`` and must resolve inside
    ``output_dir``. This renderer never launches a server or runs a simulation.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir).resolve()

    def _link(self, value):
        if value is None:
            return None
        if ":" in value or "\\" in value or value.startswith("/"):
            raise ValueError("Studio media links must be local relative paths")
        path = (self.output_dir / "studio" / value).resolve()
        if not path.is_relative_to(self.output_dir) or not path.is_file():
            raise ValueError(f"Studio link must resolve to an existing file inside the output root: {value}")
        return value

    def export(self, entries: Sequence[StudioEntry], *, theory_documents: Mapping[str, Path] | None = None) -> Path:
        if not entries or len({entry.key for entry in entries}) != len(entries):
            raise ValueError("Studio requires a nonempty catalog with unique keys")
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+", entry.key) for entry in entries):
            raise ValueError("Studio entry keys may contain only letters, digits, underscores and hyphens")
        # Resolve and project all inputs before changing the export directory.
        projected = [(entry, StudioDataset(entry.result).to_dict(), self._link(entry.data_href),
                      self._link(entry.video_href), self._link(entry.evidence_href), self._link(entry.lifecycle_href)) for entry in entries]
        figures = {entry.key: [dict(title=str(figure['title']),href=self._link(figure['href']))
                              for figure in entry.figures] for entry in entries}
        lifecycle_outcomes = {}
        for entry, dataset, raw, _, _, lifecycle in projected:
            payload = json.loads((self.output_dir / 'studio' / raw).read_text(encoding='utf-8'))
            recorded = ADTMPCMissionResult.from_dict(payload.get('result', payload)).to_dict()
            if json.dumps(recorded, sort_keys=True, allow_nan=False) != json.dumps(entry.result.to_dict(), sort_keys=True, allow_nan=False):
                raise ValueError(f"Studio raw record differs from the supplied result: {entry.key}")
            if lifecycle is not None:
                physical = json.loads((self.output_dir / 'studio' / lifecycle).read_text(encoding='utf-8'))
                expected = hashlib.sha256(json.dumps(recorded, sort_keys=True,
                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
                if (physical.get('schema') != 'orbexa.lifecycle.studio.v1'
                        or physical.get('approach_canonical_sha256') != expected
                        or physical.get('synthetic') is not dataset['synthetic']):
                    raise ValueError(f"Lifecycle replay is not bound to this approach: {entry.key}")
                contacts = physical.get('surface_contacts', {})
                if physical.get('attachment_success') and set(contacts) != {track['id'] for track in dataset['tracks']}:
                    raise ValueError(f"Lifecycle attachment record does not cover this fleet: {entry.key}")
                lifecycle_outcomes[entry.key] = dict(physical['outcome'], stages=physical.get('stages'),
                    attached_chaser_count=len(contacts) if physical.get('attachment_success') else 0)
        documents = [(title, Path(path)) for title, path in (theory_documents or {}).items()]
        document_text = [path.read_text(encoding='utf-8') for _, path in documents]
        if documents:
            try:
                import mistune
            except ImportError as exc:
                raise ImportError("Theory rendering requires the optional 'studio' dependency: pip install orbexa[studio]") from exc
        document_links = {Path(path).resolve(): f"theory-{index + 1}.html"
                          for index, (_, path) in enumerate(documents)}
        studio = self.output_dir / "studio"
        studio.mkdir(parents=True, exist_ok=True)
        assets = Path(__file__).with_name("studio")
        shutil.copytree(assets, studio, dirs_exist_ok=True)
        local_assets = sorted(path for path in assets.iterdir() if path.is_file())
        asset_version = hashlib.sha256(b"".join(path.name.encode() + b"\0" + path.read_bytes()
                                                for path in local_assets)).hexdigest()[:16]
        catalog = {"schema": "orbexa.mission.studio.catalog.v1", "asset_version": asset_version,
                   "entries": [], "theory": []}
        for entry, dataset, raw, video, evidence, lifecycle in projected:
            filename = f"mission-{entry.key}.json"
            (studio / filename).write_text(json.dumps(dataset, allow_nan=False, separators=(",", ":")), encoding="utf-8")
            catalog["entries"].append({"key": entry.key, "title": entry.title, "dataset": filename,
                "data_href": raw, "video_href": video, "evidence_href": evidence, "lifecycle_href": lifecycle,
                "physical_outcome": lifecycle_outcomes.get(entry.key),
                "figures": figures[entry.key],
                "raw_sha256": hashlib.sha256((studio / raw).read_bytes()).hexdigest()})
        for index, (title, source) in enumerate(documents):
            heading_ids = {}
            class DocumentRenderer(mistune.HTMLRenderer):
                def heading(self, text, level, **attrs):
                    plain = html.unescape(re.sub(r"<[^>]+>", "", text)).lower()
                    slug = re.sub(r"\s+", "-", re.sub(r"[^\w\s-]", "", plain)).strip("-") or "section"
                    count = heading_ids.get(slug, 0)
                    heading_ids[slug] = count + 1
                    anchor = slug if count == 0 else f"{slug}-{count}"
                    return f'<h{level} id="{html.escape(anchor, quote=True)}">{text}</h{level}>\n'

                def link(self, text, url, title=None):
                    relative, separator, fragment = url.partition("#")
                    if ":" not in relative and relative.endswith(".md"):
                        target = (Path(source).parent / relative).resolve()
                        if target not in document_links:
                            return '<span title="Document not included in this bundle">' + text + '</span>'
                        url = document_links[target] + ("#" + fragment if separator else "")
                    elif relative and ":" not in relative and not relative.startswith("//"):
                        target = (studio / relative).resolve()
                        if not target.is_relative_to(studio.parent) or not target.is_file():
                            return '<span title="Source file not included in this bundle">' + text + '</span>'
                    return super().link(text, url, title)
            markdown = mistune.create_markdown(renderer=DocumentRenderer(escape=True), plugins=["table", "math", "url"])
            body = markdown(document_text[index])
            name = f"theory-{index + 1}.html"
            document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            document += f'<title>{html.escape(title)} · ORBEX-A</title><link rel="stylesheet" href="studio.css"><link rel="stylesheet" href="vendor/katex/katex.min.css">'
            document += '<body class="theory-document"><main class="prose">' + body + '</main>'
            document += '<script defer src="vendor/katex/katex.min.js"></script><script defer src="vendor/katex/auto-render.min.js"></script><script defer src="theory.js"></script></body></html>'
            (studio / name).write_text(document, encoding="utf-8")
            catalog["theory"].append({"title": title, "href": name})
        (studio / "catalog.json").write_text(json.dumps(catalog, indent=2, allow_nan=False), encoding="utf-8")
        # A refreshed saved-data export must not retain an older cached renderer.
        # Vendor URLs and their pinned registry bytes remain unchanged.
        names = {path.name for path in local_assets}
        for path in [studio / name for name in names if name.endswith('.js')] + list(studio.glob('*.html')):
            content = path.read_text(encoding='utf-8')
            if path.suffix == '.js':
                content = re.sub(r"(\b(?:from|import)\s*['\"])(\./[^'\"?]+\.js)(['\"])",
                    lambda match: f"{match[1]}{match[2]}?v={asset_version}{match[3]}"
                    if match[2][2:] in names else match[0], content)
            else:
                content = re.sub(r"(\b(?:src|href)=['\"])([^'\"/?]+\.(?:js|css))(['\"])",
                    lambda match: f"{match[1]}{match[2]}?v={asset_version}{match[3]}"
                    if match[2] in names else match[0], content)
            path.write_text(content, encoding='utf-8')
        return studio / "index.html"
