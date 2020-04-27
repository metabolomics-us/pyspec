from pyspec.loader.mona import MoNALoader, MoNAQueryGenerator


def test_load_record():
    mona = MoNALoader()
    data = mona.load_spectrum('AU101801')

    assert(data.id == 'AU101801')
    assert(len(data.properties) == 5)

def test_query_by_splash():
    mona = MoNALoader()
    query = MoNAQueryGenerator().query_by_splash('splash10-0a4i-3910000000-a82697559a690c6121ea')

    data = mona.query(query)

    assert(len(data) >= 1)
    assert(any(s.id == 'AU101801' for s in data))

def test_query_by_inchikey():
    mona = MoNALoader()
    queries = [
        MoNAQueryGenerator().query_by_inchikey('SLMGETRBIHOOMX-ROPABARZNA-N'),
        MoNAQueryGenerator().query_by_partial_inchikey('SLMGETRBIHOOMX')
    ]

    for query in queries:
        data = mona.query(query)

        # positive and negative mode
        assert(len(data) >= 2)
        assert(set(s.properties['ionization mode'] for s in data if 'ionization mode' in s.properties) == {'positive', 'negative'})

def test_query_by_metadata():
    mona = MoNALoader()
    query = MoNAQueryGenerator().query_by_metadata('instrument type', 'in-silico QTOF')

    data = mona.query(query, page_size=5)

    assert(len(data) >= 1)
    assert(all(s.id.startswith('LipidBlast') or s.id.startswith('FAHFA') for s in data))
