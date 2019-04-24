import json
import logging
import os
import urllib
from string import Template
from typing import List, Dict

import networkx as nx
import requests
from rdflib import URIRef, RDF

from utils import ResourceDictionary, serialize, deserialize

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)


def get_uris_of_class(repository: str, endpoint: str, sparql_file: str, class_name: str, endpoint_type: str,
                      limit: int = 1000) -> List[URIRef]:
    """
    Returns the list of uris of type class_name
    :param repository: The repository containing the RDF data
    :param endpoint: The SPARQL endpoint
    :param sparql_file: The file containing the SPARQL query
    :param class_name: The class_name to search
    :param endpoint_type: GRAPHDB or VIRTUOSO (to change the way the endpoint is called)
    :param limit: The sparql query limit
    :return: The list of uris of type class_name
    """
    uri_list = []
    uris_of_class_sparql_query = open(sparql_file).read()
    uris_of_class_template = Template(uris_of_class_sparql_query).substitute(class_name=class_name)
    uris_of_class_template = Template(uris_of_class_template + " limit $limit offset $offset ")
    for uri in get_sparql_results(uris_of_class_template, "uri", endpoint, repository,
                                  endpoint_type, limit):
        uri_list.append(uri)
        if len(uri_list) % 1000 == 0:
            print(len(uri_list))

    return uri_list


def get_properties_dict(serialized_file: str, sparql_file: str, repository: str, endpoint: str, endpoint_type: str,
                        limit: int = 1000) -> ResourceDictionary:
    """
    Return a ResourceDictionary with the list of properties in the ontology
    :param serialized_file: The file where the properties ResourceDictionary is serialized
    :param sparql_file: The file containing the SPARQL query
    :param repository: The repository containing the ontology
    :param endpoint: The SPARQL endpoint
    :param endpoint_type: GRAPHDB or VIRTUOSO (to change the way the endpoint is called)
    :param limit: The sparql query limit
    :return: A ResourceDictionary with the list of properties in the ontology
    """
    global_properties_dict = deserialize(serialized_file)
    if global_properties_dict:
        return global_properties_dict

    global_properties_dict = ResourceDictionary()
    global_properties_dict.add(RDF.type)
    properties_sparql_query = open(sparql_file).read()
    properties_sparql_query_template = Template(properties_sparql_query + " limit $limit offset $offset ")
    for rdf_property in get_sparql_results(properties_sparql_query_template, ["property"], endpoint, repository,
                                           endpoint_type, limit):
        global_properties_dict.add(rdf_property[0])

    serialize(global_properties_dict, serialized_file)
    return global_properties_dict


def get_classes_dict(serialized_file: str, sparql_file: str, repository: str, endpoint: str, endpoint_type: str,
                     limit: int = 1000) -> ResourceDictionary:
    """
    Return a ResourceDictionary with the list of classes in the ontology
    :param serialized_file: The file where the properties ResourceDictionary is serialized
    :param sparql_file: The file containing the SPARQL query
    :param repository: The repository containing the ontology
    :param endpoint: The SPARQL endpoint
    :param endpoint_type: GRAPHDB or VIRTUOSO (to change the way the endpoint is called)
    :param limit: The sparql query limit
    :return: A ResourceDictionary with the list of classes in the ontology
    """
    classes_dictionary = deserialize(serialized_file)
    if classes_dictionary:
        return classes_dictionary
    classes_dictionary = ResourceDictionary()
    classes_sparql_query = open(sparql_file).read()
    classes_sparql_query_template = Template(classes_sparql_query + " limit $limit offset $offset ")
    for class_uri in get_sparql_results(classes_sparql_query_template, ["class"], endpoint, repository,
                                        endpoint_type, limit):
        classes_dictionary.add(class_uri[0])

    serialize(classes_dictionary, serialized_file)
    return classes_dictionary


def get_properties_groups(serialized_file: str, sparql_file: str, repository: str, endpoint: str, endpoint_type: str,
                          properties_dict: ResourceDictionary,
                          limit: int = 1000) -> Dict:
    """
    Return a dictionary containing the group ids for each property in the ontology (The group ids are determined via connected components)
    :param serialized_file: The file where the properties ResourceDictionary is serialized
    :param sparql_file: The file containing the SPARQL query
    :param repository: The repository containing the ontology
    :param endpoint: The SPARQL endpoint
    :param endpoint_type: GRAPHDB or VIRTUOSO (to change the way the endpoint is called)
        :param properties_dict: The ResourceDictionary containing the properties of the ontology

    :param limit: The sparql query limit
    :return: A dictionary containing the group ids for each property
    """
    if os.path.isfile(serialized_file):
        properties_groups = deserialize(serialized_file)
        return properties_groups
    encoding_dir = os.path.dirname(serialized_file)
    if not os.path.exists(encoding_dir):
        os.makedirs(encoding_dir)

    sub_properties_dict = {}
    get_sub_properties_query = open(sparql_file).read()
    get_sub_properties_query_template = Template(get_sub_properties_query + " limit $limit offset $offset ")
    for (property1, property2) in get_sparql_results(get_sub_properties_query_template, ["property1", "property2"],
                                                     endpoint, repository,
                                                     endpoint_type, limit):
        if property2 not in sub_properties_dict:
            sub_properties_dict[property2] = []

        sub_properties_dict[property2].append(property1)

    G = nx.Graph()
    for property1 in sub_properties_dict:
        for property2 in sub_properties_dict[property1]:
            G.add_edge(property1, property2)
    for property_uri in properties_dict:
        G.add_node(property_uri)
    properties_connected_components = {}
    index = 0
    for c in nx.connected_components(G):
        for p in c:
            properties_connected_components[p] = index
        index += 1

    serialize(properties_connected_components, serialized_file)
    return properties_connected_components


def get_sparql_results(query_template, variables, endpoint, repository, endpoint_type, limit=1000):
    more_results = True
    offset = 0
    try:
        while more_results:
            sparql_query = query_template.substitute(offset=str(offset), limit=limit)
            if endpoint_type == "GRAPHDB":
                sparql_results = graphdb_query(sparql_query, repository, endpoint)
            else:
                sparql_results = sparqlQuery(sparql_query, endpoint)
            if len(sparql_results) < limit:
                more_results = False
            for result in sparql_results:
                yield [URIRef(result[variable]['value']) for variable in variables]
            offset += limit
    except:
        logging.error(
            "SPARQL query error. Please make sure the ontology is loaded in repository %s at %s or change config",
            repository, endpoint)
        exit()


def sparqlQuery(query, baseURL, format="application/json"):
    params = {
        "default-graph": "",
        "should-sponge": "soft",
        "query": query,
        "debug": "on",
        "timeout": "",
        "format": format,
        "save": "display",
        "fname": ""
    }
    querypart = urllib.parse.urlencode(params).encode("utf-8")
    response = urllib.request.urlopen(baseURL, querypart).read()
    json_response = json.loads(response)
    return json_response['results']['bindings']


def graphdb_query(query, repository, baseURL="http://localhost:7200/repositories/", limit=1000, debug=False, offset=0,
                  infer=True, sameAs=True,
                  verbatim=False):
    headers = {'Accept': 'application/json,*/*;q=0.9',
               'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
               'X-Requested-With': 'XMLHttpRequest',
               'X-GraphDB-Repository': repository}

    params = {
        "infer": infer,
        "offset": offset,
        "query": query,
        "limit": limit,
        "sameAs": sameAs,
    }
    response = requests.post(baseURL + repository, headers=headers, params=params)
    if debug:
        print(response.text)
    if verbatim:
        return response.text
    json_response = json.loads(response.text)
    return json_response['results']['bindings']


def graphdb_ask(query, repository, baseURL="http://localhost:7200/repositories/", infer=True, sameAs=False):
    headers = {'Accept': 'application/sparql-results+json,*/*;q=0.9',
               'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
               'X-Requested-With': 'XMLHttpRequest',
               'X-GraphDB-Repository': repository}

    params = {
        "infer": infer,
        "query": query,
        "sameAs": sameAs,
    }
    response = requests.post(baseURL + repository, headers=headers, params=params)
    print(response)
    print(response.text)
    json_response = json.loads(response.text)
    return bool(json_response['boolean'])
