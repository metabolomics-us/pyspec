import argparse
import collections
import json
import os
import tqdm

from typing import List

from pyspec.msms_spectrum import MSMSSpectrum


def load_libraries(libraries: List[str]):
    data = []

    for library in libraries:
        if not os.path.exists(library):
            print(f'{library} does not exist!')
            continue

        if library.lower().endswith('.json'):
            data.extend(load_json_library(library))
        else:
            print(f'{library} uses an unsupported library format!')

    # Transform spectrum and sort library data by precursor m/z
    data = [MSMSSpectrum(x['spectrum'], record_id=x['id'], name=x['name'], precursor_mz=x['precursor'], inchikey=x['inchikey']) for x in data]
    data.sort(key=lambda x: x.precursor)

    return data

def load_json_library(filename: str):
    """
    A JSON library is expected to contain an array of objects consisting of:
      * id (accession or record ID)
      * name
      * inchikey
      * spectrum (MoNA/SPLASH formatted spectrum string)
      * precursor (precursor m/z)
      * adduct (optional, to be utilized further in the future)
    """

    with open(filename) as f:
        print(f'Loading library {filename}')
        data = json.load(f)

        # Minimal validation
        assert(all('id' in x for x in data))
        assert(all('spectrum' in x for x in data))
        assert(all('precursor' in x for x in data))

        return data


def load_dataset(filename: str):
    """
    Assumes a CSV data table or a JSON format similar to the JSON library, requiring only an
    id, spectrum and precursor
    """

    if filename.lower().endswith('.json'):
        with open(filename) as f:
            print(f'Loading dataset {filename}')
            data = json.load(f)

    elif filename.endswith('.csv'):
        df = pd.read_csv(filename)
        data = []

        for i, row in df.iterrows():
            data.append({
                'id': row['id'],
                'spectrum': row['spectrum'],
                'precursor': row['id']
            })

    else:
        print(f'{filename} uses an unsupported data format!')
        data = []

    # Minimal validation
    assert(all('id' in x for x in data))
    assert(all('spectrum' in x for x in data))
    assert(all('precursor' in x for x in data))

    data = [MSMSSpectrum(x['spectrum'], record_id=x['id'], precursor_mz=x['precursor']) for x in data]
    data.sort(key=lambda x: x.precursor)
    return data


def msms_similarity_search(dataset, library, ms1_tolerance, ms2_tolerance, match_threshold, hide_progress=True):
    results = []
    lib_start = 0

    if not hide_progress:
        print(f'Searching {len(dataset)} against {len(library)} library spectra...')

    for s in tqdm.tqdm(dataset, disable=hide_progress):
        best_match = None
        best_match_sim = 0

        for i in range(lib_start, len(library)):
            lib = library[i]

            # Limit search to precursor tolerance
            if lib.precursor < s.precursor - ms1_tolerance:
                lib_start = i
                continue
            if lib.precursor > s.precursor + ms1_tolerance:
                break

            # Perform similarity matching
            cosine_sim, reverse_sim, presence_sim, _, _, total_sim = s.total_similarity(lib, ms1_tolerance, ms2_tolerance)

            if total_sim > best_match_sim:
                best_match = lib
                best_match_sim = total_sim

        if best_match_sim >= match_threshold:
            results.append({
                'id': s.record_id,
                'lib_id': best_match.record_id,
                'lib_name': best_match.name,
                'lib_inchikey': best_match.inchikey,
                'mz_diff': s.precursor - lib.precursor,

                'cosine_sim': cosine_sim,
                'reverse_sim': reverse_sim,
                'presence_sim': presence_sim,
                'total_sim': total_sim
            })

    if not hide_progress:
        print(f'Annotated {len(results)}/{len(dataset)} features')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='annotates a dataset using precursor m/z and MS/MS similarity')
    parser.add_argument('dataset', help='')
    parser.add_argument('libraries', nargs='+', help='')
    parser.add_argument('--ms1_tolerance', type=float, default=0.01, help='precursor m/z matching tolerance')
    parser.add_argument('--ms2_tolerance', type=float, default=0.05, help='MS/MS similarity tolerance')
    parser.add_argument('--similarity_threshold', type=float, default=0.8, help='total MS/MS similarity threshold (0-1)')
    parser.add_argument('-o', '--output', required=True, help='output file for results')
    args = parser.parse_args()

    # Load data
    libraries = load_libraries(args.libraries)
    dataset = load_dataset(args.dataset)

    results = msms_similarity_search(dataset, libraries, args.ms1_tolerance, args.ms2_tolerance, args.similarity_threshold, hide_progress=False)

    with open(args.output, 'w') as fout:
        print(json.dumps(results, indent=2), file=fout)
