import math



def transform_spectrum_tuple(spectrum):
    s = [tuple(map(float, x.split(':'))) for x in spectrum.split()]
    return [x for x in s if x[1] > 0]


def transform_spectrum(spectrum, nominal_bin=True, normalize=True, scale_function=None):
    if type(spectrum) == str:
        spectrum = [x.split(':') for x in spectrum.split()]

    bins = collections.defaultdict(float)

    for mz, intensity in spectrum:
        key = int(float(mz) + MZ_ROUND_CORRECTION) if nominal_bin else float(mz)
        bins[key] += float(intensity)

    max_intensity = max(bins.values())

    transformed_spectrum = collections.defaultdict(float)

    for k, v in bins.items():
        if v > 0:
            transformed_spectrum[k] = v
             
            if normalize:
                transformed_spectrum[k] = 100 * transformed_spectrum[k] / max_intensity

            if scale_function is not None:
                transformed_spectrum[k] = scale_function(k, transformed_spectrum[k])

    return transformed_spectrum


def cosine_similarity(a, b):
    if type(a) in [str, list]:
        a = transform_spectrum(a)
    if type(b) in [str, list]:
        b = transform_spectrum(b)

    normA = sum(v * v for v in a.values())
    normB = sum(v * v for v in b.values())

    if normA == 0 or normB == 0:
        return 0

    sharedIons = sorted(set(a.keys()) & set(b.keys()))
    product = math.pow(sum(a[k] * b[k] for k in sharedIons), 2)

    return product / normA / normB

def composite_similarity(a, b):
    if type(a) == str:
        a = transform_spectrum(a)
    if type(b) == str:
        b = transform_spectrum(b)
    
    sharedIons = sorted(x for x in set(a.keys()) & set(b.keys()) if a[x] > EPS_CORRECTION and b[x] > EPS_CORRECTION)
    cosineSimilarity = cosine_similarity(a, b)

    if len(sharedIons) > 1:
        ratiosA = [a[x] / a[y] for x, y in zip(sharedIons, sharedIons[1:])]
        ratiosB = [b[x] / b[y] for x, y in zip(sharedIons, sharedIons[1:])]
        combinedRatios = [x / y for x, y in zip(ratiosA, ratiosB)]

        intensitySimilarity = 1 + sum(x if x < 1 else 1 / x for x in combinedRatios)

        return (len(a) * cosineSimilarity + intensitySimilarity) / (len(a) + len(sharedIons))

    else:
        return cosineSimilarity
