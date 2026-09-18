"""
fetch_sources.py
Download the commentary texts listed in sources/manifest.toml.

The texts are not part of this repository. They are fetched from archive.org
into data/sources/<key>/ (gitignored), and each file must match the sha256
pinned in the manifest -- the section tables there refer to one exact version
of each text, so a different one is refused rather than ingested.

Usage:
    python fetch_sources.py                         # every source
    python fetch_sources.py --source marx-1895      # one source
    python fetch_sources.py --force                 # re-download even if cached
"""

import sys

import click

from commentary.fetch import fetch_file
from commentary.manifest import SOURCES_DIR, load_manifest


@click.command()
@click.option("--source", "source_keys", multiple=True, help="Only fetch this source (repeatable)")
@click.option("--force", is_flag=True, help="Re-download files that are already cached")
def main(source_keys: tuple[str, ...], force: bool):
    sources = load_manifest()
    if source_keys:
        unknown = set(source_keys) - {s.key for s in sources}
        if unknown:
            raise click.BadParameter(f"not in the manifest: {', '.join(sorted(unknown))}")
        sources = [s for s in sources if s.key in source_keys]

    failed = False
    for source in sources:
        click.echo(f"▶  {source.key} — {source.citation()}")
        click.echo(f"   {source.rights}")
        for file in source.files:
            result = fetch_file(source.url(file), source.directory() / file.name,
                                file.sha256, force=force)
            if result.ok:
                click.echo(f"   ✓ {file.name} ({result.status})")
            else:
                failed = True
                click.echo(f"   ✗ {file.name}: sha256 {result.sha256[:16]}… does not match the "
                           f"pinned {result.expected[:16]}… — kept as {file.name}.rejected, "
                           f"not ingested")
        click.echo()

    click.echo(f"Texts are in {SOURCES_DIR.relative_to(SOURCES_DIR.parent.parent)}/ (not tracked by git).")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
