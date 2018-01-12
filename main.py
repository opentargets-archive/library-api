from __future__ import division

import os
import re
from collections import Counter, deque, defaultdict
from copy import deepcopy
from datetime import datetime
from difflib import SequenceMatcher
import community
import jmespath
import networkx as nx
import numpy as np
import requests
from addict import Dict
from elasticsearch import Elasticsearch, helpers, RequestsHttpConnection
from elasticsearch_xpack import XPackClient
from flask import Flask, request, jsonify, render_template, url_for
from flask import Markup
from flask_cors import CORS
from flask_env import MetaFlaskEnv
from networkx.readwrite import json_graph
from rope.base.codeanalyze import ChangeCollector
from BioStopWords import DOMAIN_STOP_WORDS
from bio_lexicon import lex as lex_dict


# def is_gae():
#     if 'SERVER_SOFTWARE' in os.environ and (os.environ['SERVER_SOFTWARE'].startswith('Google App Engine/') or
#                                                 os.environ['SERVER_SOFTWARE'].startswith('Development/')):
#         from google.appengine.api import urlfetch
#         urlfetch.set_default_fetch_deadline(60)
#
#         return True
#
# if is_gae():
#     import requests_toolbelt.adapters.appengine
#     # Use the App Engine Requests adapter. This makes sure that Requests uses
#     # URLFetch.
#     requests_toolbelt.adapters.appengine.monkeypatch()

class Configuration(object):
    __metaclass__ = MetaFlaskEnv

    ENV_PREFIX = 'LIBRARY_'


NONWORDCHARS_REGEX = re.compile(r'\W+', flags=re.IGNORECASE | re.UNICODE)


ES_MAIN_URL = os.environ["ES_MAIN_URL"].split(',')
ES_GRAPH_URL = os.environ["ES_GRAPH_URL"].split(',')

PUB_INDEX_LAMBDA = 'pubmed-18'
PUB_DOC_TYPE = 'publication'

PUB_INDEX = 'pubmed-18'
# PUB_DOC_TYPE = 'publication'

CONCEPT_INDEX = 'pubmed-18-concept'
CONCEPT_DOC_TYPE = 'concept'

MARKED_INDEX = 'pubmed-18-taggedtext'
MARKED_DOC_TYPE = 'taggedtext'

BIOENTITY_INDEX = 'pubmed-18-bioentity'
BIOENTITY_DOC_TYPE = 'bioentity'

es_graph = Elasticsearch(ES_GRAPH_URL,
                         # sniff_on_start=True,
                         # connection_class=RequestsHttpConnection
                         )
xpack = XPackClient(es_graph)

xpack.info()

es = Elasticsearch(ES_MAIN_URL,
                   timeout=300,
                   # sniff_on_start=True,
                   # connection_class=RequestsHttpConnection
                   )

app = Flask(__name__)
app.config.from_object(Configuration)
CORS(app)

gene_names = []

session = requests.Session()


FUILTERED_NODES = ['antibody',
              'function',
              'phenotype',
              'Mice',
              'CROMOLYN SODIUM',
              'sign or symptom',
              'AMINO ACIDS',
              'GLYCINE',
              'SODIUM CHLORIDE',
              'WATER',
              'RASAGILINE',
              'FELBINAC',
              'TCHP',
              'Moderate',
              'Sporadic',
              'Positive',
              'Negative',
              'Chronic',
              'Position',
                   'Borderline',
                   'Profound',
                   'Laterality',
                   'Bilateral',
                   'Unilateral',
                   'Right',
                   'Right-sided',
                   'Left',
                   'Left-sided',
                   'Generalized',
                   'Generalised',
                   'Localized',
                   'Localised',
                   'Distal',
                   'Outermost',
                   'Proximal',
                   'Severity',
                   'Intensity',
                   'Mild',
                   'Severe',
                   'Frequency',
                   'Frequent',
                   'Episodic',
                   'Acute',
              'Chronic',
              'Heterogeneous',
              'Prolonged',
              'disease',
              'cancer',
              'cancers',
              'neoplasm',
              'neoplasms',
              'tumor',
              'tumors',
              'tumour',
              'tumours',
              'Central',
              'Onset',
              'Refractory',
              'Progressive',
              'target',
              'Spastic paraplegia - epilepsy - intellectual disability'
                   ]
FUILTERED_NODES.extend(DOMAIN_STOP_WORDS)


def _ratio(str1, str2):
    s = SequenceMatcher(None, str1, str2)
    return s.quick_ratio()



# =============== PORTED FROM TEXTACY ======================
# ported from textacy https://github.com/chartbeat-labs/textacy
# to allow code to run in google app engine and aws lambda

def token_sort_ratio(str1, str2):
    """
    Measure of similarity between two strings based on minimal edit distance,
    where ordering of words in each string is normalized before comparing.

    Args:
        str1 (str)
        str2 (str)

    Returns:
        float: similarity between ``str1`` and ``str2`` in the interval [0.0, 1.0],
            where larger values correspond to more similar strings.
    """
    if not str1 or not str2:
        return 0
    str1 = _force_unicode(str1)
    str2 = _force_unicode(str2)
    str1_proc = _process_and_sort(str1)
    str2_proc = _process_and_sort(str2)
    return _ratio(str1_proc, str2_proc)


def aggregate_term_variants(terms,
                            acro_defs=None,
                            fuzzy_dedupe=True):
    """
    Take a set of unique terms and aggregate terms that are symbolic, lexical,
    and ordering variants of each other, as well as acronyms and fuzzy string matches.

    Args:
        terms (Set[str]): set of unique terms with potential duplicates
        acro_defs (dict): if not None, terms that are acronyms will be
            aggregated with their definitions and terms that are definitions will
            be aggregated with their acronyms
        fuzzy_dedupe (bool): if True, fuzzy string matching will be used
            to aggregate similar terms of a sufficient length

    Returns:
        List[Set[str]]: each item is a set of aggregated terms

    Notes:
        Partly inspired by aggregation of variants discussed in
        Park, Youngja, Roy J. Byrd, and Branimir K. Boguraev.
        "Automatic glossary extraction: beyond terminology identification."
        Proceedings of the 19th international conference on Computational linguistics-Volume 1.
        Association for Computational Linguistics, 2002.
    """
    agg_terms = []
    seen_terms = set()
    for term in sorted(terms, key=len, reverse=True):

        if term in seen_terms:
            continue

        variants = set([term])
        seen_terms.add(term)

        # symbolic variations
        if '-' in term:
            variant = term.replace('-', ' ').strip()
            if variant in terms.difference(seen_terms):
                variants.add(variant)
                seen_terms.add(variant)
        if '/' in term:
            variant = term.replace('/', ' ').strip()
            if variant in terms.difference(seen_terms):
                variants.add(variant)
                seen_terms.add(variant)

        # lexical variations
        term_words = term.split()
        # last_word = term_words[-1]
        # # assume last word is a noun
        # last_word_lemmatized = lemmatizer.lemmatize(last_word, 'n')
        # # if the same, either already a lemmatized noun OR a verb; try verb
        # if last_word_lemmatized == last_word:
        #     last_word_lemmatized = lemmatizer.lemmatize(last_word, 'v')
        # # if at least we have a new term... add it
        # if last_word_lemmatized != last_word:
        #     term_lemmatized = ' '.join(term_words[:-1] + [last_word_lemmatized])
        #     if term_lemmatized in terms.difference(seen_terms):
        #         variants.add(term_lemmatized)
        #         seen_terms.add(term_lemmatized)

        # if term is an acronym, add its definition
        # if term is a definition, add its acronym
        if acro_defs:
            for acro, def_ in acro_defs.items():
                if acro.lower() == term.lower():
                    variants.add(def_.lower())
                    seen_terms.add(def_.lower())
                    break
                elif def_.lower() == term.lower():
                    variants.add(acro.lower())
                    seen_terms.add(acro.lower())
                    break

        # if 3+ -word term differs by one word at the start or the end
        # of a longer phrase, aggregate
        if len(term_words) > 2:
            term_minus_first_word = ' '.join(term_words[1:])
            term_minus_last_word = ' '.join(term_words[:-1])
            if term_minus_first_word in terms.difference(seen_terms):
                variants.add(term_minus_first_word)
                seen_terms.add(term_minus_first_word)
            if term_minus_last_word in terms.difference(seen_terms):
                variants.add(term_minus_last_word)
                seen_terms.add(term_minus_last_word)
            # check for "X of Y" <=> "Y X" term variants
            if ' of ' in term:
                split_term = term.split(' of ')
                variant = split_term[1] + ' ' + split_term[0]
                if variant in terms.difference(seen_terms):
                    variants.add(variant)
                    seen_terms.add(variant)

        # intense de-duping for sufficiently long terms
        if fuzzy_dedupe is True and len(term) >= 13:
            for other_term in sorted(terms.difference(seen_terms), key=len, reverse=True):
                if len(other_term) < 13:
                    break
                tsr = token_sort_ratio(term, other_term)
                if tsr > 0.93:
                    variants.add(other_term)
                    seen_terms.add(other_term)
                    break

        agg_terms.append(variants)

    return agg_terms


def _force_unicode(s):
    """Force ``s`` into unicode, or die trying."""
    if isinstance(s, unicode):
        return s
    else:
        return unicode(s)


def _process_and_sort(s):
    """Return a processed string with tokens sorted then re-joined."""
    return ' '.join(sorted(_process(s).split()))


def _process(s):
    """
    Remove all characters but letters and numbers, strip whitespace,
    and force everything to lower-case.
    """
    if not s:
        return ''
    return NONWORDCHARS_REGEX.sub(' ', s).lower().strip()
# =============== END  TEXTACY ======================



def agf_opt_merge(lists):
    """merge lists
    https://stackoverflow.com/questions/9110837/python-simple-list-merging-based-on-intersections
    agf (optimized)
    """
    sets = deque(set(lst) for lst in lists if lst)
    results = []
    disjoint = 0
    if sets:
        current = sets.pop()
        while True:
            merged = False
            newsets = deque()
            for _ in range(disjoint, len(sets)):
                this = sets.pop()
                if not current.isdisjoint(this):
                    current.update(this)
                    merged = True
                    disjoint = 0
                else:
                    newsets.append(this)
                    disjoint += 1
            if sets:
                newsets.extendleft(sets)
            if not merged:
                results.append(current)
                try:
                    current = newsets.pop()
                except IndexError:
                    break
                disjoint = 0
            sets = newsets
        return results
    return sets



# =============== FLASK METHODS ======================


@app.after_request
def add_header(response):
    response.cache_control.max_age = 300
    return response



@app.route('/document/<string:doc_id>', methods=['GET'])
def document_get(doc_id):
    document = session.get('/'.join((ES_MAIN_URL[0], PUB_INDEX, PUB_DOC_TYPE, doc_id)),
                           ).json()

    return jsonify(document)


@app.route('/document-more-like-this/<string:doc_id>', methods=['GET'])
def document_mlt(doc_id):
    query_body = {
        "query": {
            "more_like_this": {
                "fields": ["title", "abstract", "keywords", "mesh_headings.label"],
                "like": [
                    {
                        "_index": PUB_INDEX,
                        "_type": PUB_DOC_TYPE,
                        "_id": doc_id
                    },
                ],
                "min_term_freq": 1,
                "max_query_terms": 25
            }
        }
    }

    data = session.post(ES_MAIN_URL[0] + '/' + PUB_INDEX + '/_search/',
                        json=query_body
                        ).json()

    return jsonify(data)


@app.route('/search', methods=['GET'])
def search_get():
    query = request.args.get('query')
    size = request.args.get('size') or 10
    sort = request.args.get('sort') or [{"pub_date": "desc"}]
    search_after = request.args.get('search_after') or []
    search_after_id = request.args.get('search_after_id')
    aggs = request.args.get('aggs')
    if not isinstance(search_after, list):
        search_after = [search_after]
    if search_after_id:
        search_after.append(search_after_id)
        sort.append({"pub_id": "desc"})
    query_body = {
        "query": {
            "query_string": {
                "query": query
            },
        },
        "size": size,
        "sort": sort,
        # "_source": False,
        "_source": {
            "excludes": ["*_sentences", 'filename', 'text_mined*']
        },
        # "highlight": {
        #     "fields": {
        #         "*": {},
        #     },
        #     "require_field_match": False,
        # }
    }

    if aggs:
        query_body['aggregations'] = {  # "abstract_significant_terms": {
            #     "significant_terms": {"field": "abstract"}
            # },
            # "title_significant_terms": {
            #     "significant_terms": {"field": "title"}
            # },
            # "mesh_headings_label_significant_terms": {
            #     "significant_terms": {"field": "mesh_headings.label"}
            # },
            # "keywords_significant_terms": {
            #     "significant_terms": {"field": "keywords"}
            # },
            "journal_abbr_significant_terms": {
                "significant_terms": {"field": "journal.medlineAbbreviation"}
            },
            # "chemicals_name_significant_terms": {
            #     "significant_terms": {"field": "chemicals.name"}
            # },
            "authors_significant_terms": {
                "significant_terms": {"field": "authors.short_name"}
            },
            "pub_date_histogram": {
                "date_histogram": {
                    "field": "pub_date",
                    "interval": "year"
                }
            },
            "acronym_significant_terms": {
                "significant_terms": {"field": "text_mined_entities.nlp.named_entities",
                                      # "mutual_information": {
                                      #     "include_negatives": False
                                      # },
                                      "size": 20

                                      },
            },
            "genes": {
                "significant_terms": {
                    "field": "text_mined_entities.nlp.tagged_entities_grouped.GENE|OPENTARGETS.reference",
                    "size": 20,
                    "mutual_information": {}

                },
                "aggs": {

                    "gene_hits": {
                        "top_hits": {

                            "_source": {
                                "includes": ["text_mined_entities.nlp.tagged_entities_grouped.GENE|OPENTARGETS.label",
                                             "text_mined_entities.nlp.tagged_entities_grouped.GENE|OPENTARGETS.reference"]
                            },
                            "size": 5
                        }
                    }
                }
            },
            "diseases": {
                "significant_terms": {
                    "field": "text_mined_entities.nlp.tagged_entities_grouped.DISEASE|OPENTARGETS.reference",
                    "size": 20,
                    "mutual_information": {}

                },

                "aggs": {

                    "disease_hits": {
                        "top_hits": {

                            "_source": {
                                "includes": [
                                    "text_mined_entities.nlp.tagged_entities_grouped.DISEASE|OPENTARGETS.label",
                                    "text_mined_entities.nlp.tagged_entities_grouped.DISEASE|OPENTARGETS.reference"]
                            },
                            "size": 5
                        }
                    }
                }
            },
            "drugs": {
                "significant_terms": {
                    "field": "text_mined_entities.nlp.tagged_entities_grouped.DRUG|CHEMBL.reference",
                    "size": 20,
                    "mutual_information": {}

                },
                "aggs": {

                    "drug_hits": {
                        "top_hits": {

                            "_source": {
                                "includes": ["text_mined_entities.nlp.tagged_entities_grouped.DRUG|CHEMBL.label",
                                             "text_mined_entities.nlp.tagged_entities_grouped.DRUG|CHEMBL.reference"]
                            },
                            "size": 5
                        }
                    }
                }
            },
            "phenotypes": {
                "significant_terms": {
                    "field": "text_mined_entities.nlp.tagged_entities_grouped.PHENOTYPE|HPO.reference",
                    "size": 20,
                    "mutual_information": {}
                },
                "aggs": {

                    "phenotype_hits": {
                        "top_hits": {

                            "_source": {
                                "includes": ["text_mined_entities.nlp.tagged_entities_grouped.PHENOTYPE|HPO.label",
                                             "text_mined_entities.nlp.tagged_entities_grouped.PHENOTYPE|HPO.reference"]
                            },
                            "size": 5
                        }
                    }
                }
            },
            "top_chunks_significant_terms": {
                "significant_terms": {"field": "text_mined_entities.nlp.top_chunks",
                                      "mutual_information": {
                                          "include_negatives": False
                                      },
                                      "size": 20

                                      },
                "aggs": {
                    # "sample": {
                    #     # "diversified_sampler": {
                    #     #     "shard_size": 100,
                    #     #     "field": "text_mined_entities.nlp.abbreviations.short"
                    #     # },
                    #     "sampler": {
                    #              "shard_size": 500000,
                    #             },
                    # "aggs": {
                    "top_chunk_hits": {
                        "top_hits": {
                            "sort": [
                                {
                                    "text_mined_entities.nlp.abbreviations.long": {
                                        "order": "desc"
                                    }
                                }
                            ],
                            "_source": {
                                "includes": ["text_mined_entities.nlp.abbreviations",
                                             "text_mined_entities.nlp.acronyms"]
                            },
                            "size": 10
                        }
                    }
                    # }

                    # },
                    # "chunks_significant_terms": {
                    #     "significant_terms": {"field": "text_mined_entities.noun_phrases.chunks",
                    #                           "jlh": {
                    #                               "include_negatives": False
                    #                             }
                    #                           }
                    # },

                }
            }
        }
    if search_after:
        query_body['search_after'] = search_after

    data = session.post(ES_MAIN_URL[0] + '/' + PUB_INDEX + '/_search/',
                        json=query_body
                        ).json()
    if '_shards' in data:
        del data['_shards']
    if 'aggregations' in data:
        # if data['aggregations']['acronym_significant_terms']['buckets']:
        #     new_gene_buckets = []
        #     # genes = set(json.load(open('/Users/andreap/work/code/pubmine/human_gene_names_hgnc_lower.json')))
        #     acronyms = set([i['key'].lower() for i in data['aggregations']['acronym_significant_terms']['buckets']])
        #     gene_acronyms = acronyms #  & genes
        #
        #     for bucket in data['aggregations']['acronym_significant_terms']['buckets']:
        #         if bucket['key'].lower() in gene_acronyms:
        #             new_gene_buckets.append(bucket)
        #
        #     data['aggregations']['gene_significant_terms'] = dict(buckets=new_gene_buckets)

        terms = []
        term_weight = defaultdict(lambda: 0)
        abbreviations = {}
        inverted_abbreviations = {}
        for i in data['aggregations']['top_chunks_significant_terms']['buckets']:
            terms.append(i['key'])
            term_weight[i['key']] = i['bg_count']
            for hit in i['top_chunk_hits']['hits']['hits']:
                for a in hit['_source']['text_mined_entities']['nlp']['abbreviations']:
                    short_name = a['short'].lower()
                    long_name = a['long'].lower()
                    abbreviations[short_name] = long_name
                    if long_name not in inverted_abbreviations:
                        inverted_abbreviations[long_name] = []
                    if short_name not in inverted_abbreviations[long_name]:
                        inverted_abbreviations[long_name].append(short_name)
        'merge similar abbreviations'
        synonyms = {}
        for acronyms in inverted_abbreviations.values():
            if len(acronyms) > 1:
                for acronym in acronyms:
                    if acronym not in synonyms:
                        synonyms[acronym] = []
                    # synonyms[acronym].extend(acronyms)
                    synonyms[acronym] = list((set(synonyms[acronym]) | set(acronyms)))
                    # for pair in list(itertools.combinations(list(set(acronyms)), 2)):
                    #     if pair[0]!=pair[1]:
                    #         synonyms[pair[0]]=pair[1]
        for k, v in synonyms.items():
            v.pop(v.index(k))
        # pprint(synonyms)
        for t in terms:
            if t.lower() in abbreviations:
                terms.append(abbreviations[t.lower()])
        # pprint(terms)
        groups = aggregate_term_variants(set(terms), abbreviations)
        groups_with_syn = []
        for g in groups:
            new_g = deepcopy(g)
            for i in g:
                if i in synonyms:
                    new_g = new_g | set(synonyms[i])
            groups_with_syn.append(new_g)
        # pprint(groups_with_syn)
        groups_merge = agf_opt_merge(groups_with_syn)
        # pprint(groups_merge)
        filtered_terms = {max(list(group), key=lambda x: term_weight[x]): group for group in groups_merge}
        # pprint(filtered_terms)
        filtered_agg = []
        for bucket in data['aggregations']['top_chunks_significant_terms']['buckets']:
            if bucket['key'] in filtered_terms:
                bucket.pop('top_chunk_hits')
                bucket['alternative_terms'] = list(filtered_terms[bucket['key']] - set([bucket['key']]))
                filtered_agg.append(bucket)
        data['aggregations']['top_chunks_significant_terms']['buckets'] = filtered_agg[:15]

        for i in data['aggregations']['genes']['buckets']:
            for hit in i['gene_hits']['hits']['hits']:
                for entity in hit['_source']['text_mined_entities']['nlp']['tagged_entities_grouped'][
                    'GENE|OPENTARGETS']:
                    if entity['reference'] == i['key']:
                        i['label'] = entity['label'].replace('_', ' ')
                        break
            del i['gene_hits']
        for i in data['aggregations']['diseases']['buckets']:
            for hit in i['disease_hits']['hits']['hits']:
                for entity in hit['_source']['text_mined_entities']['nlp']['tagged_entities_grouped'][
                    'DISEASE|OPENTARGETS']:
                    if entity['reference'] == i['key']:
                        i['label'] = entity['label'].replace('_', ' ')
                        break
            del i['disease_hits']
        for i in data['aggregations']['drugs']['buckets']:
            for hit in i['drug_hits']['hits']['hits']:
                for entity in hit['_source']['text_mined_entities']['nlp']['tagged_entities_grouped']['DRUG|CHEMBL']:
                    if entity['reference'] == i['key']:
                        i['label'] = entity['label'].replace('_', ' ')
                        break
            del i['drug_hits']
        for i in data['aggregations']['phenotypes']['buckets']:
            for hit in i['phenotype_hits']['hits']['hits']:
                for entity in hit['_source']['text_mined_entities']['nlp']['tagged_entities_grouped']['PHENOTYPE|HPO']:
                    if entity['reference'] == i['key']:
                        i['label'] = entity['label'].replace('_', ' ')
                        break
            del i['phenotype_hits']

    # if data['aggregations']['acronym_significant_terms']['buckets']:
    #     new_gene_buckets = []
    #     genes = set(json.load(open('/Users/andreap/work/code/pubmine/human_gene_names_hgnc_lower.json')))
    #     acronyms = set([i['key'].lower() for i in data['aggregations']['acronym_significant_terms']['buckets']])
    #     gene_acronyms = acronyms & genes
    #
    #     for bucket in data['aggregations']['acronym_significant_terms']['buckets']:
    #         if bucket['key'].lower() in gene_acronyms:
    #             new_gene_buckets.append(bucket)
    #
    #     data['aggregations']['gene_significant_terms'] = dict(buckets=new_gene_buckets)

    return jsonify(data)


@app.route('/graph/explore', methods=['POST'])
def graph_proxy_post():
    data = session.post(ES_GRAPH_URL[0] + '/' + PUB_INDEX_LAMBDA + '/_xpack/_graph/_explore', json=request.json).json()
    if data and 'vertices' in data and data['vertices']:
        data, topics = get_topics_from_graph(data)
        return jsonify(dict(graph=data,
                            topics=topics))
    return jsonify({})
    # return jsonify(xpack.graph.explore(index=PUB_INDEX,
    #                            body=request.json))


def build_graph(data):
    '''
    builds a networkX graph from a response of the elasticsearch Ghraph API
    :param data:
    :return:
    '''
    if data and data['vertices']:
        '''build graph'''
        G = nx.Graph()
        node_names = {}
        node_class = []
        node_classes = []
        gene_nodes = []
        for i, vertex in enumerate(data['vertices']):
            G.add_node(i,
                       term=vertex['term'],
                       weight=vertex['weight'],
                       flat_weight=1.
                       )
            if vertex['term'].lower() in gene_names:
                gene_nodes.append(i)
            node_names[i] = vertex['term']
            if vertex['field'] not in node_classes:
                node_classes.append(vertex['field'])
            node_class.append(node_classes.index(vertex['field']))

        edge_sizes = []
        for i, edge in enumerate(data['connections']):
            G.add_edge(edge['source'],
                       edge['target'],
                       # weight=edge['weight'],
                       flat_weight=1.,
                       doc_count=edge['doc_count'])
            edge_sizes.append(edge['doc_count'])

        return G


@app.route('/search/topic', methods=['GET'])
def topic_graph():
    index_name = request.args.get('index') or PUB_INDEX_LAMBDA
    query = request.args.get('query')
    verbose = request.args.get('verbose') or False
    result_count = 0
    count_query = {"query": {"query_string": {"query": query}},
                   "size": 0}
    r = session.post(ES_GRAPH_URL[0] + '/' + index_name + '/_search/', json=count_query)
    if r.ok:
        result_count = r.json()['hits']['total']
    if result_count > 0:
        if result_count < 50:
            field_default = ["text_mined_entities.nlp.chunks"]
            size_default = 50
            specificity_Default = 1
            sample_size_default = 1000
            timeout_default = 1000
        elif result_count < 100:
            field_default = ["text_mined_entities.nlp.top_chunks"]
            size_default = 100
            specificity_Default = 2
            sample_size_default = 2000
            timeout_default = 3000
        elif result_count < 500:
            field_default = ["text_mined_entities.nlp.top_chunks"]
            size_default = 100
            specificity_Default = 3
            sample_size_default = 2000
            timeout_default = 3000
        else:
            field_default = ["text_mined_entities.nlp.top_chunks"]
            size_default = 200
            specificity_Default = 5
            sample_size_default = 5000
            timeout_default = 10000
        size = request.args.get('size') or size_default
        specificity = request.args.get('specificity') or specificity_Default
        field = request.args.getlist('field') or field_default
        if len(field) == 1 and isinstance(field[0], (str, unicode)):
            field = field[0].split(',')

        vertices_query = []
        for f in field:
            vertices_query.append({"field": f, "min_doc_count": int(specificity), "size": int(size)})
        es_query = {"query": {"query_string": {"query": query}},
                    "controls": {"use_significance": True, "sample_size": sample_size_default,
                                 "timeout": timeout_default},
                    "connections": {"vertices": vertices_query},
                    "vertices": vertices_query}
        r = session.post(ES_GRAPH_URL[0] + '/' + index_name + '/_xpack/_graph/_explore', json=es_query)
        if r.ok:
            data = r.json()
            if data and 'vertices' in data and data['vertices']:
                data, topics = get_topics_from_graph(data, verbose)
                return jsonify(dict(graph=data,
                                    topics=topics))
    return jsonify({})


def get_topics_from_graph(data, verbose=False, min_topic_size=1):
    G = build_graph(data)
    # first compute the best partition
    node_terms = nx.get_node_attributes(G, 'term')
    node_weights = nx.get_node_attributes(G, 'weight')
    node_centralities = nx.degree_centrality(G)
    partition = community.best_partition(G, weight='flat_weight', resolution=.33)
    topics = []
    count = 0
    nodes_to_delete = []
    for topic_id, com in enumerate(set(partition.values())):
        count += 1.
        com_nodes = [nodes for nodes in partition.keys()
                     if partition[nodes] == com]
        if len(com_nodes) > min_topic_size:
            topic = []
            for node in com_nodes:
                topic.append((node_centralities[node], node_terms[node], node))
                data['vertices'][node]['topic'] = topic_id
            topic = sorted(topic, reverse=True)
            subtopics = dict(top=[],
                             total=0)
            subtopic_list = [dict(centrality=node_centralities[i[2]],
                                  topic_label=node_terms[i[2]],
                                  weight=node_weights[i[2]],
                                  vertex=i[2],
                                  topic=topic_id
                                  ) for i in topic[1:]]
            if verbose:
                subtopics = dict(top=[i for i in subtopic_list[0:10]],
                                 total=len(subtopic_list))
            else:
                subtopics = dict(top=[i['vertex'] for i in subtopic_list[0:10]],
                                 total=len(subtopic_list))
            topics.append(dict(topic_label=topic[0][1],
                               connected_topics=subtopics,
                               weight=node_weights[topic[0][2]],
                               centrality=node_centralities[topic[0][2]],
                               vertex=topic[0][2],
                               topic=topic_id
                               )
                          )
        else:
            nodes_to_delete.extend(com_nodes)

    if nodes_to_delete:

        nodes_to_delete = sorted(nodes_to_delete)
        new_node_map = {}
        offset = 0
        for i, node in enumerate(data['vertices']):
            if i in nodes_to_delete:
                offset += 1
            else:
                new_node_map[i] = i - offset

        data['vertices'] = [v for i, v in enumerate(data['vertices']) if i not in nodes_to_delete]

        new_connections = []
        for c in data['connections']:
            if c['source'] not in nodes_to_delete and c['target'] not in nodes_to_delete:
                c['source'] = new_node_map[c['source']]
                c['target'] = new_node_map[c['target']]
                new_connections.append(c)
        data['connections'] = new_connections

        for topic in topics:
            topic['vertex'] = new_node_map[topic['vertex']]



            # if 'topic' not in node:
            #     print i,nodes_to_delete.index(i)+1
            # del data['vertices'][]

    topics = sorted(topics, key=lambda x: (x['centrality'], x['weight']), reverse=True)
    if not verbose:
        for t in topics:
            del t['centrality']
            del t['weight']
            del t['topic_label']
    return data, topics





def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    # print np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    return np.degrees((np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))))


@app.route('/trends')
def trends2():
    query = request.args.get('query')
    years_to_consider = 15
    query_body = {
        "query": {
            "bool": {
                "must": [
                    {"query_string": {"query": query}},
                    {"range": {
                        "pub_date": {"gte": "now-%sy/y" % years_to_consider,
                                     "le": "now-1y/y"}
                    }
                    }
                ]
            }
        },
        "aggs": {
            # "sample": {
            # "diversified_sampler": {
            #     "shard_size": 300000,
            #     "field" : "journal.medlineAbbreviation"#TODO: index the first and the last authors as single value
            # to allow for a proper diversification here
            # },
            # "sampler": {
            #          "shard_size": 500000,
            #         },
            # "aggs": {
            "docs_per_year": {
                "date_histogram": {
                    "field": "pub_date",
                    "interval": "year",
                    "format": "yyyy",
                    "keyed": True
                },

                "aggs": {
                    "global_moving_average": {
                        "moving_avg": {"buckets_path": "_count",
                                       "window": 3,
                                       "model": "holt",
                                       "settings": {
                                           "alpha": .8,
                                           "beta": 0.3
                                       },
                                       "predict": 2
                                       }
                    }
                }
            },
            "entities": {
                "terms": {
                    "field": "text_mined_entities.noun_phrases.top_chunks",
                    # "jlh": {
                    #     "include_negatives": False,
                    # },
                    # "min_doc_count":2*years_to_consider,
                    # "background_filter": {
                    #         "query_string": {"query": query},
                    #     },
                    "size": 1000,
                },
                "aggs": {
                    "years": {
                        "date_histogram": {
                            "field": "pub_date",
                            "interval": "year",
                            "format": "yyyy",
                            "keyed": True
                        },
                        "aggs": {

                            "moving_average": {
                                "moving_avg": {"buckets_path": "_count",
                                               "window": 5,
                                               "model": "holt",
                                               "settings": {
                                                   "alpha": 0.3,
                                                   "beta": 0.3
                                               },
                                               "predict": 2
                                               }
                            },
                            # "deriv": {
                            #     "derivative": {
                            #         "buckets_path": "doc_count"
                            #     }
                            # },
                            # "2nd_deriv": {
                            #     "derivative": {
                            #         "buckets_path": "deriv"
                            #     }
                            # },
                            # "serial_diff": {
                            #     "buckets_path": "doc_count",
                            #     # "lag": "1"
                            # },
                            #     "surprise": {
                            #         "bucket_script": {
                            #             "buckets_path": {
                            #                 "doc_count": "sample>entities>years>moving_average",
                            #                 "docs_per_year": "sample>docs_per_year>global_moving_average"
                            #             },
                            #             "script": "(params.doc_count / params.docs_per_year)"
                            #         }
                        }
                    }
                    # },
                    # "largest_surprise": {
                    #     "max_bucket": {
                    #         "buckets_path": "years.surprise"
                    #     }
                    # }
                },
            },
            # "ninetieth_surprise": {
            #     "percentiles_bucket": {
            #         "buckets_path": "entities>largest_surprise",
            #         "percents": [
            #             90.0
            #         ]
            #     }
            # }
            # }
            # }
        },
        "size": 0

    }

    r = session.post(ES_MAIN_URL[0] + '/' + PUB_INDEX + '/_search/',
                     json=query_body
                     )
    data = r.json()
    if r.ok:
        output = {  # 'zdata': data,
        }
        novel_trends = {}
        established_trends = {}
        negative_trends = {}
        debug = []
        recency_year = 3
        recency_year_index = -(2 + recency_year)  # don't consider the current year
        new_trends_threshold_year = datetime.now().year - recency_year - 1
        try:
            docs_per_year = jmespath.search('aggregations.docs_per_year.buckets', data)
            docs_per_year_as_dict = []
            global_moving_average = []
            for year, counts in sorted(docs_per_year.items()):
                if 'global_moving_average' in counts:
                    global_moving_average.append(counts['global_moving_average']['value'])
                    docs_per_year_as_dict.append({year: counts['global_moving_average']['value']})
            global_moving_average_array = np.array(global_moving_average)
            output['docs_per_year'] = docs_per_year_as_dict
            for term_data in jmespath.search('aggregations.entities.buckets[*]', data):
                term_moving_average = []
                term_moving_average_ids = []
                term_counts = []
                term_counts_asdict = []
                years_with_data = []
                for year, counts in sorted(term_data['years']['buckets'].items()):
                    normalised_counts = counts['doc_count']
                    years_with_data.append(year)
                    # if docs_per_year[year]['doc_count'] and counts['doc_count']:
                    #     normalised_counts = counts['doc_count']/docs_per_year[year]['doc_count']
                    term_counts_asdict.append({year: normalised_counts})
                    term_counts.append(normalised_counts)
                    if 'moving_average' in counts:
                        term_moving_average.append(counts['moving_average']['value'])  # /docs_per_year[year][
                        # 'global_moving_average']['value'])
                        term_moving_average_ids.append(year)
                term_counts_array = np.array(term_counts)
                moving_average_array = np.array(term_moving_average)
                moving_average_gradient = np.gradient(moving_average_array)  # /term_moving_average
                moving_average_gradient_2 = np.gradient(moving_average_gradient)
                # if moving_average_gradient_2.std()/moving_average_gradient_2.mean() >0.2 :#and
                # moving_average_gradient_2.sum() >0:
                moving_average_gradient_asdict = []
                moving_average_gradient_2nd_asdict = []
                term_moving_average_asdict = []
                for i in range(len(term_moving_average_ids)):
                    moving_average_gradient_asdict.append({term_moving_average_ids[i]: moving_average_gradient[i]})
                    moving_average_gradient_2nd_asdict.append(
                        {term_moving_average_ids[i]: moving_average_gradient_2[i]})
                    term_moving_average_asdict.append({term_moving_average_ids[i]: term_moving_average[i]})
                recency_year = 3
                recency_year_index = -(2 + recency_year)  # don't consider the current year
                recent_counts = term_counts_array[recency_year_index:-2]  # don't consider the current year
                previous_counts = term_counts_array[:recency_year_index]  # don't consider the current year
                if len(previous_counts):
                    recent_gradient = moving_average_gradient[recency_year_index - 2:-1].sum() / previous_counts[
                        0]  # sum the gradient of the last 5 years, escluding the current one and the two predicted ones
                    recent_gradient_all = (
                        np.array([i for i in moving_average_gradient[recency_year_index - 2:-1]]) > 0).all()
                    recent_gradient_2nd = moving_average_gradient_2[recency_year_index - 2:-1].sum()
                    recent_gradient_2nd_all = (
                        np.array([i for i in moving_average_gradient_2[recency_year_index - 2:-1]]) > 0).all()
                    total_gradient = moving_average_gradient.sum()
                    total_gradient_trend = moving_average_gradient.mean() / term_counts_array.mean()
                    total_gradient_2nd = moving_average_gradient_2.sum()
                    total_gradient_2nd_std = moving_average_gradient_2[:-1].std()
                    total_gradient_2nd_avg = np.median(moving_average_gradient_2[:-1])
                    spike_gradient_2nd = moving_average_gradient_2.max() / moving_average_gradient_2.mean()
                    prediction = moving_average_gradient[-3] / term_moving_average[
                        -1]  # (term_moving_average[-1]-term_moving_average[-2])
                    trend_start_treshold = max(term_counts) / 4.
                    trend_start = years_with_data[0]
                    for counts in term_counts_asdict:
                        if counts.values()[0] >= trend_start_treshold:
                            trend_start = int(counts.keys()[0])
                            break

                    if np.any(previous_counts):
                        recent_increase_ratio_avg = np.mean(recent_counts) / np.mean(previous_counts)  # ratio of the
                        # last 3 year over the maxum 3 years before that
                        recent_increase_ratio_max = np.max(recent_counts) / np.max(
                            previous_counts)  # the max value of the last 3 years compared to the max value before that
                    else:
                        recent_increase_ratio_avg = np.sum(recent_counts)
                        recent_increase_ratio_max = np.max(recent_counts)

                    local_global_moving_average_array = np.array(
                        [i.values()[0] for i in docs_per_year_as_dict if i.keys()[0] in term_moving_average_ids])
                    angle_between_trends = angle_between(moving_average_array, local_global_moving_average_array)
                    # debug.append((angle_between_trends,term_data['key'], (angle_between_trends-15)))#,
                    # recent_counts, previous_counts ))
                    if abs(angle_between_trends) > 15:
                        corrected_prediction = ((prediction + (
                            recent_increase_ratio_avg + recent_increase_ratio_max / 2) / 10) + \
                                                total_gradient_trend + \
                                                (recent_counts[-1] / previous_counts[0]) / 10 + \
                                                total_gradient_2nd_avg) * (angle_between_trends / 30.)
                        coorrected_negative_prediction = -prediction
                        if corrected_prediction > 0 \
                                and recent_increase_ratio_avg > 1 \
                                and recent_increase_ratio_max >= 1 \
                                and np.sum(term_counts_array) > len(term_counts_array) * 2 \
                                and unicode(datetime.now().year + 1) in years_with_data:

                            if trend_start >= (new_trends_threshold_year):
                                # if len(previous_counts)<=new_trends_threshold_year-recency_year:
                                # print 'NOVEL:',term_data['key']
                                novel_trends[term_data['key']] = dict(
                                    prediction=corrected_prediction,
                                    term_moving_average=term_moving_average_asdict,
                                    term_counts=term_counts_asdict,
                                    trend_start=trend_start
                                )



                            else:
                                # print 'ESTABLISHED:', term_data['key']
                                established_trends[term_data['key']] = dict(
                                    prediction=corrected_prediction,
                                    term_moving_average=term_moving_average_asdict,
                                    term_counts=term_counts_asdict,
                                    trend_start=trend_start

                                )

                        elif coorrected_negative_prediction > 0 \
                                and np.sum(term_counts_array) > len(term_counts_array) * 2:
                            negative_trends[term_data['key']] = dict(
                                prediction=coorrected_negative_prediction,
                                term_moving_average=term_moving_average_asdict,
                                term_counts=term_counts_asdict,
                            )

            output['novel_trends'] = []
            sorted_novel_trends = sorted([(v['prediction'], k) for k, v in novel_trends.items()], reverse=True)
            for p, k in sorted_novel_trends[:30]:
                output['novel_trends'].append({k: novel_trends[k]})
            output['total_novel'] = len(sorted_novel_trends)

            output['established_trends'] = []
            sorted_established_trends = sorted([(v['prediction'], k) for k, v in established_trends.items()],
                                               reverse=True)
            for p, k in sorted_established_trends[:30]:
                output['established_trends'].append({k: established_trends[k]})
            output['total_established'] = len(sorted_established_trends)

            output['negative_trends'] = []
            sorted_negative_trends = sorted([(v['prediction'], k) for k, v in negative_trends.items()], reverse=True)
            for p, k in sorted_negative_trends[:30]:
                output['negative_trends'].append({k: negative_trends[k]})
            output['total_negative'] = len(sorted_negative_trends)



        except KeyError as e:
            print 'keyerror', e
            pass
        return jsonify(output)
    return jsonify(data)




def digest_buckets_data(buckets,
                        reference_query,
                        label_query):
    for bucket in buckets:
        label = bucket['key']
        if 'ENSG00000114062' in bucket['key']:
            pass
        if 'label' in bucket:
            top_hits = bucket['label']
            if top_hits['hits']['hits']:
                for hit in top_hits['hits']['hits']:
                    hit_ids = jmespath.search(reference_query, hit)
                    hit_labels = jmespath.search(label_query, hit)
                    for i, hit_id in enumerate(hit_ids):
                        if hit_id == bucket['key']:
                            label = hit_labels[i]
                            break
        bucket['label'] = label


def is_entity_in_subject(conc, query):
    if 'subject_tags' in conc:
        for subject_tag_categories in conc['subject_tags']:
            for subject_tag in conc['subject_tags'][subject_tag_categories]:
                if subject_tag['reference'] == query:
                    return True
    return False


def get_entities_in_concept(conc, valid_ids):
    return_dict = dict(subject=set(),
                       object=set())
    if 'subject_tags' in conc:
        for subject_tag_categories in conc['subject_tags']:
            for subject_tag in conc['subject_tags'][subject_tag_categories]:
                if subject_tag['reference'] in valid_ids:
                    return_dict['subject'].add(subject_tag['reference'])
    if 'object_tags' in conc:
        for object_tag_categories in conc['object_tags']:
            for object_tag in conc['object_tags'][object_tag_categories]:
                if object_tag['reference'] in valid_ids:
                    return_dict['object'].add(object_tag['reference'])

    return_dict['subject'] = list(return_dict['subject'])
    return_dict['object'] = list(return_dict['object'])

    return return_dict



def digest_matrix_key(key):
    second = None
    splitted_key = key.split('&')
    first = splitted_key.pop(0)
    relation = 'sig_term'
    if first.startswith('sbj'):
        relation = 'sbj'
        first = first[3:]
    elif first.startswith('obj'):
        relation = 'obj'
        first = first[3:]

    if splitted_key:
        second = splitted_key.pop(0)
        if second.startswith('sbj'):
            relation += '-sbj'
            second = second[3:]
        elif second.startswith('obj'):
            relation += '-obj'
            second = second[3:]

    return first, second, relation


def get_base_entity_query(query,
                          minimum_should_match,
                          query_type="cross_fields"):
    return {

        "multi_match": {
            "query": query,
            "fields": ["concept.subject_tags.*",
                       "concept.object_tags.*",
                       # "concept.sentence.text",# this will trigger a big number of hits
                       'pub_id',
                       "abbreviations.*"
                       ],
            "analyzer": "whitespace",
            "minimum_should_match": minimum_should_match,
            "type": query_type,
        },
    }


def get_sbj_obj_agg(query,
                    elements_count=5,
                    sampler_size=0,
                    minimum_should_match=1,
                    mode='specificity',
                    min_doc_count=1,
                    entities=None,
                    sampler_type='standard',
                    query_type='cross_fields'
                    ):
    if entities is None:
        entities = ['DISEASE',
                    'GENE',
                    'DRUG',
                    'CONCEPT',
                    'PHENOTYPE']
    q = Dict()
    q.size = 0
    q.query = get_base_entity_query(query, minimum_should_match, query_type)
    count_weights = defaultdict(lambda: 1)
    count_weights['PHENOTYPE'] = 0.8
    # count_weights['DISEASE'] = 0.8
    # count_weights['CONCEPT'] = .5#TODO: put to 1.5 once concepts are proper entities
    aggs = Dict()
    data_aggs = Dict()
    semtype2tagindex = dict(subj='subject_tags', obj='object_tags')
    if mode == 'popularity':
        for entity_type in entities:
            for sem_type in ['subj', 'obj']:
                # if entity_type == 'CONCEPT':
                #     data_aggs[entity_type.lower() + '_' + sem_type] = {
                #         "terms": {
                #             "field": "concept." + sem_type + "ect.keyword",
                #             "size": elements_count,
                #             "min_doc_count": min_doc_count
                #
                #         },
                #
                #     }
                # else:
                data_aggs[entity_type.lower() + '_' + sem_type] = {
                    "terms": {
                        "field": "concept.%s.%s.reference" % (semtype2tagindex[sem_type], entity_type),
                        "size": int(elements_count * count_weights[entity_type]),
                        "min_doc_count": min_doc_count,
                        "exclude": FUILTERED_NODES
                    },
                    "aggs": {

                        "label": {
                            "top_hits": {

                                "_source": {
                                    "includes": [
                                        "concept.%s.%s.label" % (semtype2tagindex[sem_type], entity_type),
                                        "concept.%s.%s.reference" % (semtype2tagindex[sem_type], entity_type)]
                                },
                                "size": 3
                            }
                        }
                    }
                }
    elif mode == 'specificity':
        for entity_type in entities:
            for sem_type in ['subj', 'obj']:
                # if entity_type == 'CONCEPT':
                #     data_aggs[entity_type.lower() + '_' + sem_type] = {
                #         "significant_terms": {
                #             "field": "concept." + sem_type + "ect.keyword",
                #             "size": int(elements_count * count_weights[entity_type]),
                #             "min_doc_count": min_doc_count,
                #             "chi_square": {"include_negatives": False}
                #             # "gnd": {}
                #         },
                #
                #     }
                # else:
                data_aggs[entity_type.lower() + '_' + sem_type] = {
                    "significant_terms": {
                        "field": "concept.%s.%s.reference" % (semtype2tagindex[sem_type], entity_type),
                        "size": int(elements_count * count_weights[entity_type]),
                        "min_doc_count": min_doc_count,
                        "exclude": FUILTERED_NODES,
                        "chi_square": {"include_negatives": False}
                        # "gnd": {}

                    },
                    "aggs": {

                        "label": {
                            "top_hits": {

                                "_source": {
                                    "includes": [
                                        "concept.%s.%s.label" % (semtype2tagindex[sem_type], entity_type),
                                        "concept.%s.%s.reference" % (semtype2tagindex[sem_type], entity_type)]
                                },
                                "size": 3
                            }
                        }
                    }
                }

    if sampler_size:
        if sampler_type == 'diversified':
            aggs.sample.diversified_sampler = {
                "shard_size": sampler_size,
                "max_docs_per_value": 3,
                "field": "concept.subject"
                # "script" : {
                #     "lang": "painless",
                #     "source": "doc['concept.subject.keyword'].value+doc['concept.object.keyword'].value"
                # }
            }
        else:
            aggs.sample.sampler.shard_size = sampler_size

        aggs.sample.aggs = data_aggs.to_dict()
    else:
        aggs = data_aggs

    q.aggs = aggs.to_dict()
    # print json.dumps(q, indent=2, sort_keys=True)
    return q.to_dict()


def get_adj_matrix_query(valid_ids,
                         sampler_size=0,
                         max_nodes=250):
    get_edges_data = {
        "multi_match": {
            "query": ' '.join([key for key, value in valid_ids.items()]),  # if value['category'] != 'CONCEPT']),
            "fields": ["concept.subject_tags.*",
                       "concept.object_tags.*",
                       # "concept.subject.keyword",
                       # "concept.object.keyword",
                       ],
            # "operator": "and",
            "analyzer": "whitespace",
            "minimum_should_match": 2,
            "type": "cross_fields",
            # "cutoff_frequency": .05
        },

    }

    matrix_filters = {}
    if len(valid_ids) > max_nodes:
        app.logger.error('TOO MANY NODES: cannot find edges for these nodes %i %s' % (
            len(valid_ids.keys()[max_nodes:]), valid_ids.keys()[max_nodes:]))
        for i in valid_ids.keys()[max_nodes:]:
            valid_ids.pop(i)

    for key in valid_ids.keys()[:max_nodes]:
        if valid_ids[key]['category']:
            # if valid_ids[key]['category'] == 'CONCEPT':
            #     matrix_filters['sbj' + key] = {
            #         "match": {
            #             "concept.subject.keyword": {
            #                 "query": key,
            #                 "operator": "and"
            #             }
            #         }
            #     }
            #     matrix_filters['obj' + key] = {
            #         "match": {
            #             "concept.object.keyword": {
            #                 "query": key,
            #                 "operator": "and"
            #             }
            #         }
            #     }
            # else:
            matrix_filters['sbj' + key] = {
                "terms": {"concept.subject_tags.%s.reference" % valid_ids[key]['category']: [key]}}
            matrix_filters['obj' + key] = {"terms": {
                "concept.object_tags.%s.reference" % valid_ids[key]['category']: [key]}}

    if sampler_size:
        adj_matrix_query = {
            "query": get_edges_data,
            "size": 0,
            "aggs": {
                "sample": {
                    # "sampler": {
                    #     "shard_size": sampler_size
                    # },
                    "diversified_sampler": {
                        "shard_size": sampler_size,
                        "max_docs_per_value": 10,
                        "field": "concept.subject"
                        # "script": {
                        #     "lang": "painless",
                        #     "source": "doc['concept.subject.keyword'].value+doc['concept.object.keyword'].value"
                        # }
                    },
                    "aggs": {
                        "edges": {
                            "adjacency_matrix": {
                                "filters": matrix_filters
                            }
                        }
                    }}
            },
            "timeout": '5m'
        }
    else:
        adj_matrix_query = {
            "query": get_edges_data,
            "size": 0,
            "aggs": {
                "edges": {
                    "adjacency_matrix": {
                        "filters": matrix_filters
                    }
                }
                # }}
            },
            "timeout": '5m'
        }
    return adj_matrix_query


def clean_valid_ids(valid_ids, valid_ids_lables):
    '''remove concepts with same label as entity and phenotypes with same label as disease'''
    valid_ids_lower = [i.lower() for i in valid_ids.keys()]
    disease_labels = []
    for ent in valid_ids.values():
        if ent['category'] == 'DISEASE':
            disease_labels.append(ent['label'].lower())
            disease_labels.append(ent['label'].lower().replace('-', ''))
    for k, v, in valid_ids.items():
        if v['category'] == 'CONCEPT':
            lower_label = v['label'].lower()
            if lower_label in valid_ids_lables or \
                            v['label'] in valid_ids_lables or \
                            lower_label.replace('-', '') in valid_ids_lables:
                valid_ids.pop(k)
            if v['label'][-1] == 's' and \
                            v['label'][:-1] in valid_ids:
                valid_ids.pop(k)
                # if v['label'] != lower_label and lower_label in valid_ids_lower and k in valid_ids:
                #     valid_ids.pop(k)
        elif v['category'] == 'PHENOTYPE' and disease_labels:
            if v['label'].lower() in disease_labels or \
                            v['label'].lower().replace('-', '') in disease_labels:
                valid_ids.pop(k)


@app.route('/data', methods=['GET'])
def entity_map():
    index_name = request.args.get('index') or CONCEPT_INDEX
    query = request.args.get('query')
    query = query.replace(',', ' ')
    default_entity_types = 'DISEASE|DRUG|GENE|PHENOTYPE|CONCEPT'.split('|')
    entity_types = request.args.get('entity_type')
    if entity_types is not None:
        if '|' in entity_types:
            entity_types = entity_types.split('|')
        elif ',' in entity_types:
            entity_types = entity_types.split(',')
        else:
            entity_types=[entity_types]
        if entity_types[0] not in default_entity_types:
            entity_types = default_entity_types
    else:
        entity_types = default_entity_types
    # if 'CONCEPT' not in entity_types:
    #     entity_types.append('CONCEPT')
    elments_count = int(request.args.get('elements', 5)) or 5
    min_doc_count = request.args.get('min_doc_count') or 0
    sampler_size = int(request.args.get('sample', 0)) or 0

    allowed_modes = ['specificity', 'popularity']
    mode = request.args.get('mode', 'specificity') or 'specificity'
    score_model = request.args.get('score', 'pagerank') or 'pagerank'
    if mode not in allowed_modes:
        mode = 'pagerank'
    if mode == 'popularity':
        score_model = 'pagerank'

    if not min_doc_count:
        if sampler_size:
            total_base_docs = sampler_size * 5
        else:
            count_docs = es.search(index=index_name,
                                   body={'query': get_base_entity_query(query,
                                                                        len(query.split(' ')),
                                                                        query_type='cross_fields'),
                                         'size': 0})
            total_base_docs = count_docs['hits']['total']
        if total_base_docs > 1000000:  # force sampler if query too big
            if not sampler_size:
                sampler_size = int(total_base_docs / 20)
                min_doc_count = 5
        if total_base_docs > 500000:  # force sampler if query too big
            if not sampler_size:
                sampler_size = int(total_base_docs / 10)
                min_doc_count = 2
        elif total_base_docs > 30000:  # increase doc count
            min_doc_count = 5
        else:
            min_doc_count = int(total_base_docs / 5000) + 1

    data = es.search(index=index_name,
                     body=get_sbj_obj_agg(query,
                                          elements_count=elments_count,
                                          sampler_size=sampler_size,
                                          mode=mode,
                                          minimum_should_match=len(query.split(' ')),
                                          min_doc_count=min_doc_count,
                                          entities=entity_types,
                                          query_type='cross_fields'),
                     timeout='5m')
    if sampler_size:
        agg_data = data['aggregations']['sample']
    else:
        agg_data = data['aggregations']

    for agg_name, agg in agg_data.items():
        if isinstance(agg, dict):
            digest_buckets_data(agg['buckets'],
                                reference_query='_source.concept.*.*[][][].reference',
                                label_query='_source.concept.*.*[][][].label')

    valid_ids = {}
    valid_ids_lables = set()
    for agg_name, agg in agg_data.items():
        if isinstance(agg, dict):
            agg_category = ''
            if 'disease' in agg_name:
                agg_category = 'DISEASE'
            elif 'phenotype' in agg_name:
                agg_category = 'PHENOTYPE'
            elif 'drug' in agg_name:
                agg_category = 'DRUG'
            elif 'gene' in agg_name:
                agg_category = 'GENE'
            elif 'concept' in agg_name:
                agg_category = 'CONCEPT'

            for bucket in agg['buckets']:
                if bucket['key'] not in valid_ids and bucket['label'] not in FUILTERED_NODES and bucket[
                    'label'].lower() not in FUILTERED_NODES:
                    if not (agg_category == 'CONCEPT' and bucket['label'].lower() in valid_ids_lables):

                        score = 1
                        if 'score' in bucket:
                            score = bucket['score']
                        valid_ids[bucket['key']] = dict(label=bucket['label'],
                                                        category=agg_category,
                                                        score=score)
                        if agg_category != 'CONCEPT':
                            valid_ids_lables.add(bucket['label'].lower())
                            valid_ids_lables.add(bucket['label'].lower().replace('-', ''))

                elif 'score' in bucket and bucket['key'] in valid_ids and bucket['label'] not in FUILTERED_NODES and bucket[
                    'label'].lower() not in FUILTERED_NODES:
                    valid_ids[bucket['key']]['score'] = (valid_ids[bucket['key']]['score'] + bucket['score']) / 2.

    clean_valid_ids(valid_ids, valid_ids_lables)

    if valid_ids:
        '''expand with additional significant_terms'''
        elments_count += int(elments_count * .5)
        query = ' '.join(valid_ids.keys())
        data = es.search(index=index_name,
                         body=get_sbj_obj_agg(query,
                                              elements_count=elments_count,
                                              sampler_size=sampler_size,
                                              minimum_should_match=len(query.split(' ')),
                                              mode=mode,
                                              min_doc_count=min_doc_count,
                                              entities=entity_types
                                              ),
                         timeout='5m')
        if sampler_size:
            agg_data = data['aggregations']['sample']
        else:
            agg_data = data['aggregations']

        for agg_name, agg in agg_data.items():
            if isinstance(agg, dict):
                digest_buckets_data(agg['buckets'],
                                    reference_query='_source.concept.*.*[][][].reference',
                                    label_query='_source.concept.*.*[][][].label')

        for agg_name, agg in agg_data.items():
            if isinstance(agg, dict):

                agg_category = ''
                if 'disease' in agg_name:
                    agg_category = 'DISEASE'
                elif 'phenotype' in agg_name:
                    agg_category = 'PHENOTYPE'
                elif 'drug' in agg_name:
                    agg_category = 'DRUG'
                elif 'gene' in agg_name:
                    agg_category = 'GENE'
                elif 'concept' in agg_name:
                    agg_category = 'CONCEPT'
                if agg_category not in ['CONCEPT']:
                    for bucket in agg['buckets']:
                        if bucket['key'] not in valid_ids and bucket['label'] not in FUILTERED_NODES and bucket[
                            'label'].lower() not in FUILTERED_NODES:
                            score = 1
                            if 'score' in bucket:
                                score = bucket['score']
                            valid_ids[bucket['key']] = dict(label=bucket['label'],
                                                            category=agg_category,
                                                            score=score)
                            if agg_category != 'CONCEPT':
                                valid_ids_lables.add(bucket['label'].lower())
                                valid_ids_lables.add(bucket['label'].lower().replace('-', ''))
                        elif 'score' in bucket and bucket['key'] in valid_ids and bucket['label'] not in FUILTERED_NODES \
                                and \
                                        bucket['label'].lower() not in FUILTERED_NODES:
                            valid_ids[bucket['key']]['score'] = (valid_ids[bucket['key']]['score'] + bucket[
                                'score']) / 2.
        clean_valid_ids(valid_ids, valid_ids_lables)

        '''query to get all the edges linking significant nodes'''

        adj_matrix_query = get_adj_matrix_query(valid_ids,
                                                sampler_size=sampler_size,
                                                max_nodes=250)
        edge_data = es.search(index=index_name,
                              body=adj_matrix_query)
        G = nx.Graph()
        for node_name, node_data in valid_ids.items():
            G.add_node(node_name,
                       label=node_data['label'],
                       category=node_data['category'],
                       significance=node_data['score'])

        edges = {}
        if sampler_size:
            edge_data_agg = edge_data['aggregations']['sample']['edges']['buckets']
        else:
            edge_data_agg = edge_data['aggregations']['edges']['buckets']
        for bucket in edge_data_agg:
            first, second, relation = digest_matrix_key(bucket['key'])
            if first and second and first != second and len(set(relation.split('-'))) > 1:

                key = '|'.join(sorted([first, second]))
                size = bucket['doc_count']
                if key not in edges:
                    edges[key] = [first, second, size]
                else:
                    edges[key][2] += size

        for edge_id, edge in edges.items():
            first, second, size = edge
            G.add_edge(first, second, size=size, id=edge_id)
            # else:
            #     print first, second, bucket['key']
        # if central_node:
        #     for node_name in valid_ids:
        #         if node_name != central_node:
        #             G.add_edge(central_node, node_name, size=1, type='sig_term')

        pagerank = nx.pagerank_numpy(G)
        # print pagerank
        for node, node_data in G.nodes(data=True):
            weight = int(round(1000. * pagerank[node], 0) + 1)
            node_data['pagerank'] = weight
            valid_ids[node]['pagerank'] = weight
            sig_weight = int(round(node_data['significance'] * 100, 0) + 1)
            node_data['significance'] = sig_weight
            valid_ids[node]['significance'] = sig_weight

        # hits = nx.hits_numpy(G)
        # print hits


        # Converts the Network graph to a JSON format
        jsonG = json_graph.node_link_data(G)
        # Fixes the network so that edges use node names instead of integers
        jsonG['edges'] = [
            {
                'source': G.nodes()[link['source']],
                'target': G.nodes()[link['target']],
                'id': link['id'],
                'size': link['size'],
            }
            for i, link in enumerate(jsonG['links'])]
        jsonG.pop('links')
        edge_table = []
        for edge in jsonG['edges']:
            try:
                row = []
                target = valid_ids[edge['target']]
                source = valid_ids[edge['source']]
                row.append('|'.join((source['label'], edge['source'], source['category'])))
                row.append('|'.join((target['label'], edge['target'], target['category'])))
                row.append(edge['size'])
                row.append(int(round((source[score_model] + target[score_model]) / 2., 0)))

                edge_table.append(row)
            except TypeError as e:
                app.logger.exception('cannot add edge to graph: '+str(edge))
        jsonG['nodes'] = sorted(jsonG['nodes'], reverse=True, key=lambda x: x[score_model])
        for key in jsonG['nodes']:
            key['size'] = key[score_model]
    else:
        edge_table = []
        jsonG = {'nodes': {},
                 'edges': {}}

    return jsonify(
        min_doc_count=min_doc_count,
        entity_type="|".join(entity_types),
        mode=mode,
        url=url_for('entity_relation_view'),
        url_home=url_for('entity_relation_ui'),
        table=edge_table,
        graph=jsonG,
    )


@app.route('/autocomplete', methods=['GET'])
def concept_autocomplete():
    query = request.args.get('query')
    categories_to_search = ['DISEASE',
                            'DISEASEALT',
                            'GENE',
                            'DRUG',
                            'CONCEPT',
                            'ANATOMY',
                            'LOC',
                            ]
    phrase_fields = ['abbreviations.long']
    keyword_fields = ['abbreviations.short']
    for category in categories_to_search:
        phrase_fields.extend(["concept.subject_tags.%s.label" % category,
                              "concept.subject_tags.%s.match" % category,
                              "concept.object_tags.%s.label" % category,
                              "concept.object_tags.%s.match" % category,
                              ])
        keyword_fields.extend(["concept.subject_tags.%s.label^5" % category,
                               "concept.subject_tags.%s.match^5" % category,
                               "concept.object_tags.%s.label^5" % category,
                               "concept.object_tags.%s.match^5" % category,
                               ])
    es_query = {
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": phrase_fields,
                            "analyzer": 'whitespace',
                            "type": "phrase_prefix",
                        }
                    },
                    {"multi_match": {
                        "query": query,
                        "fields": keyword_fields,
                        "analyzer": 'keyword',
                        # "fuzziness": "AUTO",
                        "tie_breaker": 0,
                        "type": "best_fields",
                    }
                    }
                ]
            },
        },
        '_source': ['concept.subject_tags.*',
                    'concept.object_tags.*'],
        'size': 20
    }

    res = es.search(index=CONCEPT_INDEX,
                    body=es_query)
    results = {}
    sorted_global_results = []
    if res['hits']['total']:
        for hit in res['hits']['hits']:
            tag_groups = jmespath.search('_source.concept.[*][0]', hit)

            for tag_group in tag_groups:
                for category, ents in tag_group.items():
                    if category in categories_to_search:
                        if category not in results:
                            results[category] = dict(name=category,
                                                     results=[],
                                                     result_counts=Counter(),
                                                     resultList=[])
                        for ent in ents:
                            if ent['reference'] not in results[category]['resultList']:  # TODO: return only top 3
                                results[category]['resultList'].append(ent['reference'])
                                results[category]['results'].append(dict(title=ent['reference'],
                                                                         description=ent['label']))

                            results[category]['result_counts'][ent['reference']] += 1

        for category in results:
            # print category, results[category]['result_counts'].most_common(5)
            top3 = [i[0] for i in results[category]['result_counts'].most_common(5) if i[1] > 0]
            results[category]['resultList'] = [i for i in results[category]['resultList'] if i in top3]
            results[category]['results'] = [i for i in results[category]['results'] if i['title'] in top3]
            for i in results[category]['results']:
                flat = deepcopy(i)
                flat['score'] = results[category]['result_counts'][i['title']]
                flat['category'] = category
                sorted_global_results.append(flat)
            results[category].pop('result_counts')

    sorted_global_results.sort(key=lambda x: x['score'], reverse=True)

    return jsonify(dict(results=results,
                        flatResults=sorted_global_results))


@app.route('/evidence')
def get_entity_evidence():
    allowed_categories = ['DISEASE',
                          # 'DISEASEALT',
                          'GENE',
                          'DRUG',
                          'CONCEPT',
                          # 'ANATOMY',
                          # 'LOC'
                          ]
    sbj = request.args.get('sbj')
    obj = request.args.get('obj')

    # min_should_match = 2
    # min_should_match += sbj.count(' ')
    # min_should_match += obj.count(' ')
    # get_edges_data = {
    #     "multi_match": {
    #         "query": ' '.join([sbj, obj]),
    #         "fields": ["concept.subject_tags.*.reference",
    #                    "concept.object_tags.*.reference"],
    #         # "operator": "and",
    #         "analyzer": "whitespace",
    #         "minimum_should_match": min_should_match,
    #         "type": "cross_fields",
    #     },
    #
    # }

    get_edges_data = {"match": {"concept.relations.undirected": '%s|%s' % (sbj, obj)}}
    get_all_edges_query = {
        "size": 1000,

        "query": get_edges_data,
        "sort": {"date": {"order": "desc"}},
        "_source": {"includes": [
            "pub_id",
            "date",
            "concept",

        ]}
    }
    edge_data = helpers.scan(client=es,
                             query=get_all_edges_query,
                             scroll='1h',
                             index=CONCEPT_INDEX,
                             # doc_type="search-object-disease",
                             timeout='10m',
                             preserve_order=True
                             )

    results = []
    counter = 0
    histogram = []
    year_counter = Counter()
    for counter, hit in enumerate(edge_data):
        hit = hit['_source']
        pub_date = hit['date']
        pub_id = hit['pub_id']
        concept = hit['concept']
        matches = []
        all_matches = []
        sbj_matches = []
        sbj_tags = jmespath.search('subject_tags.*', concept)
        if sbj_tags:
            for sbj_tag_group in sbj_tags:

                for tag in sbj_tag_group:
                    if tag['reference'] == sbj or tag['reference'] == obj:
                        matches.append(tag)
                        sbj_matches.append(tag)
                        all_matches.append(tag)
        obj_matches = []
        obj_tags = jmespath.search('object_tags.*', concept)
        if obj_tags:
            for obj_tag_group in obj_tags:
                for tag in obj_tag_group:
                    if tag['reference'] == sbj or tag['reference'] == obj:
                        matches.append(tag)
                        obj_matches.append(tag)
                    all_matches.append(tag)

        sbj_tag_references = set([i['reference'] for i in sbj_matches])
        obj_tag_references = set([i['reference'] for i in obj_matches])

        if sbj_tag_references and obj_tag_references:  # and not sbj_tag_references & obj_tag_references: #removes
            # duplicates but they are found by previous calls so commenteout for consistency
            verb_text = concept['verb']

            verb_score = 0
            for w in verb_text.split():
                if w in lex_dict:
                    verb_score += lex_dict[w]
            if concept['negated']:
                verb_score *= -1

            tagged_sentence = ChangeCollector(concept['sentence_text'])
            for i, tag in enumerate(
                    sorted(all_matches, key=lambda x: (x['start'], -x['end']))):
                if isinstance(tag['reference'], (list, tuple)):
                    tag_reference = tag['reference'][0]  # '|'.join(tag['reference'])
                else:
                    tag_reference = tag['reference']
                if tag['category'] in allowed_categories:
                    tagged_sentence.add_change(tag['start'], tag['start'],
                                               '<mark-%s data-entity="%s" reference-db="%s"  reference="%s">' % (
                                                   str(i), tag['category'], tag['reference_db'], tag_reference))
                    tagged_sentence.add_change(tag['end'], tag['end'], '</mark-%s>' % str(i))

            results.append(dict(pubid=pub_id,
                                date=pub_date,
                                matches=matches,
                                subjectMatches=sbj_matches,
                                objectMatches=obj_matches,
                                sentence=tagged_sentence.get_changed(),
                                sentenceNumber=concept['sentence'],
                                isNegative=concept['negated'],
                                subject=concept['subject'],
                                verb=concept['verb'],
                                object=concept['object'],
                                # verbSubTreeSentiment = 0,
                                verbSentiment=verb_score,
                                # concept=concept,
                                )
                           )
            year = int(pub_date.split('-')[0])
            if year > 1800:
                year_counter[year] += 1

    min_year = min(year_counter.keys())
    max_year = max(year_counter.keys())
    # normalisation_factor = year_counter.most_common(1)[0][1]/10.
    # for i in range(min_year, max_year+1):
    #     histogram.append([i, int(round(year_counter[i]/normalisation_factor,0))])
    # for i in range(max_year , min_year-1,  -1):
    #     histogram.append([i,year_counter[i]])
    histogram = sorted(year_counter.items(), reverse=True)
    # if res['hits']['total']:
    return jsonify(dict(results=results,
                        total=len(results),
                        linkTotal=counter,
                        histogram=histogram))



@app.route('/stats', methods=['GET'])
def stats():
    return jsonify(dict(documents=es.search(index=PUB_INDEX,
                                            size=0)['hits']['total'],
                        relations=es.search(index=CONCEPT_INDEX,
                                            size=0)['hits']['total']))


@app.route('/view', methods=['GET'])
def entity_relation_view():
    query = request.args.get('query')
    entity_type = request.args.get('entity_type')
    mode = request.args.get('mode', 'specificity')
    # data = json.loads(entity_map().data)['concepts']
    return render_template('entity_view.html',
                           data_feed=url_for('entity_map'),
                           query=query.replace(',', ' '),
                           entity_type=entity_type,
                           mode=mode)


@app.route('/')
def entity_relation_ui():
    return render_template('entity_relation_search.html',
                           form_action=url_for('entity_relation_view')
                           )


@app.route('/test/url', methods=['GET'])
def test_params():
    return jsonify({'request.args.getlist': request.args.getlist('field'),
                    'request.query_string': request.query_string,
                    'request.url': request.url})


@app.route('/entity/article', methods=['GET'])
def marked_article_ui():
    pub_id = request.args.get('pub_id')
    marked_title = marked_abstract = '<div></div>'
    if pub_id:
        entry = session.get(ES_MAIN_URL[0] + '/' + MARKED_INDEX + '/' + MARKED_DOC_TYPE + '/' + pub_id).json()
        if '_source' in entry:
            marked_title = entry['_source']['title']
            marked_abstract = entry['_source']['abstract']
    return render_template('marked_text.html',
                           form_action=url_for('marked_article_ui'),
                           marked_title=Markup(marked_title),
                           marked_abstract=Markup(marked_abstract),

                           pub_id=pub_id
                           )


@app.route('/entity/markedtext/<string:pub_id>', methods=['GET'])
def marked_article(pub_id):
    marked_title = marked_abstract = '<div class="entities"></div>'
    if pub_id:
        entry = session.get(ES_MAIN_URL[0] + '/' + MARKED_INDEX + '/' + MARKED_DOC_TYPE + '/' + pub_id).json()
        if '_source' in entry:
            marked_title = entry['_source']['title'] + '</div>'
            marked_abstract = '<div class="entities">' + entry['_source']['abstract']
    return jsonify(dict(title=Markup(marked_title),
                        abstract=Markup(marked_abstract),
                        pub_id=pub_id
                        ))

@app.route('/liveness-check',)
def liveness_check():
    return jsonify({'status':'live'})

@app.route('/readiness-check',)
def readiness_check():
    r = session.get(ES_MAIN_URL[0])
    r.raise_for_status()
    return jsonify({'status':'ready'})

if __name__ == '__main__':
    app.run()
