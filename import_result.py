"""Install only the adapter matching the evaluated A run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
FILES = (
    'adapter_config.json', 'adapter_model.safetensors', 'added_tokens.json',
    'base_identity.json', 'merges.txt', 'special_tokens_map.json',
    'tokenizer.json', 'tokenizer_config.json', 'vocab.json',
)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def install(result):
    pins = json.loads((ROOT/'artifact_identity.json').read_text())
    if sha(result) != pins['result_zip_sha256']:
        raise ValueError('Wrong RESULT.zip. Use the original A archive that is bound to the submitted metrics.')
    destination = ROOT/'weights'/'temporal_A'
    if destination.exists():
        if sha(destination/'adapter_model.safetensors') != pins['adapter_sha256']:
            raise ValueError('Existing weights do not match the evaluated run')
        return destination
    with tempfile.TemporaryDirectory(dir=destination.parent) as work, zipfile.ZipFile(result) as archive:
        stage = Path(work)/'adapter'
        stage.mkdir()
        manifest = json.loads(archive.read('RESULT_MANIFEST.json'))['files_sha256']
        for name in FILES:
            member = f'fit/adapter/{name}'
            target = stage/name
            with archive.open(member) as source, target.open('wb') as out:
                shutil.copyfileobj(source, out)
            if sha(target) != manifest[member]:
                raise ValueError('Invalid RESULT member checksum: '+member)
        if sha(stage/'adapter_model.safetensors') != pins['adapter_sha256']:
            raise ValueError('The adapter SHA differs from the score provenance')
        stage.rename(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    print(install(parser.parse_args().result))
