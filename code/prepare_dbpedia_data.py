import configparser
import json

from prepare_data_utils import get_uris_of_class, get_properties_dict, get_classes_dict, get_properties_groups
from utils import serialize, deserialize

config = configparser.ConfigParser()
with open('../config.json', 'r') as f:
    config = json.load(f)

DBPEDIA_DATA_ENDPOINT = config['DBPEDIA']['SPARQL_PARAMS']['DATA_ENDPOINT']
DBPEDIA_DATA_ENDPOINT_TYPE = config['DBPEDIA']['SPARQL_PARAMS']['DATA_ENDPOINT_TYPE']
URIS_OF_CLASS_QUERY = config['DBPEDIA']['SPARQL_PARAMS']['URIS_OF_CLASS_QUERY']
PROPERTIES_QUERY = config['DBPEDIA']['SPARQL_PARAMS']['PROPERTIES_QUERY']
QUERY_LIMIT = config['DBPEDIA']['SPARQL_PARAMS']['QUERY_LIMIT']
DBPEDIA_ONTOLOGY_ENDPOINT = config['DBPEDIA']['SPARQL_PARAMS']['ONTOLOGY_ENDPOINT']
DBPEDIA_ONTOLOGY_REPOSITORY = config['DBPEDIA']['SPARQL_PARAMS']['ONTOLOGY_REPOSITORY']
DBPEDIA_ONTOLOGY_ENDPOINT_TYPE = config['DBPEDIA']['SPARQL_PARAMS']['ONTOLOGY_ENDPOINT_TYPE']
PROPERTIES_DICT_FILE = config['DBPEDIA']['ENCODING_PARAMS']['PROPERTIES_DICT_FILE']
CLASSES_QUERY = config['DBPEDIA']['SPARQL_PARAMS']['CLASSES_QUERY']
CLASSES_DICT_FILE = config['DBPEDIA']['ENCODING_PARAMS']['CLASSES_DICT_FILE']
PROPERTIES_GROUPS_FILE = config['DBPEDIA']['ENCODING_PARAMS']['PROPERTIES_GROUPS_FILE']
SUB_PROPERTIES_QUERY = config['DBPEDIA']['SPARQL_PARAMS']['SUB_PROPERTIES_QUERY']
ENCODING_DIRECTORY = config['DBPEDIA']['ENCODING_PARAMS']['ENCODING_DIRECTORY']


def get_scientists_list(scientists_file):
    import os
    if os.path.isfile(scientists_file):
        scientists_list = deserialize(scientists_file)
        return scientists_list
    encoding_dir = os.path.dirname(scientists_file)
    if not os.path.exists(encoding_dir):
        os.makedirs(encoding_dir)
    scientists_list = get_uris_of_class("", DBPEDIA_DATA_ENDPOINT, URIS_OF_CLASS_QUERY, 'Scientist',
                                        DBPEDIA_DATA_ENDPOINT_TYPE, QUERY_LIMIT)
    serialize(scientists_list, scientists_file)
    return scientists_list


def get_dbpedia_properties():
    return get_properties_dict(ENCODING_DIRECTORY+PROPERTIES_DICT_FILE, PROPERTIES_QUERY, DBPEDIA_ONTOLOGY_REPOSITORY,
                               DBPEDIA_ONTOLOGY_ENDPOINT, DBPEDIA_ONTOLOGY_ENDPOINT_TYPE, QUERY_LIMIT)


def get_dbpedia_classes():
    return get_classes_dict(ENCODING_DIRECTORY+CLASSES_DICT_FILE, CLASSES_QUERY, DBPEDIA_ONTOLOGY_REPOSITORY, DBPEDIA_ONTOLOGY_ENDPOINT,
                            DBPEDIA_ONTOLOGY_ENDPOINT_TYPE, QUERY_LIMIT)


def get_dbpedia_properties_groups():
    dbpedia_properties = get_dbpedia_properties()
    return get_properties_groups(ENCODING_DIRECTORY+PROPERTIES_GROUPS_FILE, SUB_PROPERTIES_QUERY, DBPEDIA_ONTOLOGY_REPOSITORY,
                                 DBPEDIA_ONTOLOGY_ENDPOINT,
                                 DBPEDIA_ONTOLOGY_ENDPOINT_TYPE, dbpedia_properties, QUERY_LIMIT)
