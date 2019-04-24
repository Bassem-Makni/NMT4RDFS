import numpy as np
from rdflib import Graph


class GraphWordsDecoder:
    def __init__(self, active_properties, properties_groups, active_classes, graph_words_vocabulary):
        self.properties_groups = properties_groups
        self.active_properties = active_properties
        self.active_classes = active_classes
        self.graph_words_vocabulary = graph_words_vocabulary

    @staticmethod
    def counter_offset_id(resource_id, offset):
        if resource_id >= offset:
            resource_id = - resource_id
            resource_id += offset
            resource_id -= 1
        else:
            resource_id += 1
        return resource_id

    def id_to_resource(self, local_resources, resource_id, property_id):
        property = self.active_properties.inverse[property_id]
        property_group = self.properties_groups[property]

        if resource_id in self.active_classes.inverse:
            resource = self.active_classes.inverse[resource_id]
            return resource
        else:
            resource = local_resources[property_group].inverse[resource_id]
            return resource

    def np_to_graph_encoding(self, three_d_adjacency_matrix, offset):
        graph_encoding = {}
        non_zeros = np.nonzero(three_d_adjacency_matrix)
        properties_ids = non_zeros[0]
        subjects_ids = non_zeros[1]
        objects_ids = non_zeros[2]
        for index, property_id in enumerate(properties_ids):
            if (property_id + 1) not in graph_encoding:
                graph_encoding[property_id + 1] = []
            subject_id = self.counter_offset_id(subjects_ids[index], offset)
            object_id = self.counter_offset_id(objects_ids[index], offset)
            graph_encoding[property_id + 1].append((subject_id, object_id))
        return graph_encoding

    def graph_words_to_graph_encoding(self, graph_words_sequence):
        graph_encoding = {}
        for property_id, graph_word in enumerate(graph_words_sequence):
            if not graph_word:
                graph_encoding[property_id + 1] = []
            else:
                graph_encoding[property_id + 1] = self.graph_words_vocabulary.inverse[graph_word]
        return graph_encoding

    def decode_graph(self, encoding, resources_dict):
        graph = Graph()
        for pid in encoding:
            p = self.active_properties.inverse[pid]
            for (sid, oid) in encoding[pid]:
                s = self.id_to_resource(resources_dict, sid, pid)
                o = self.id_to_resource(resources_dict, oid, pid)
                graph.add((s, p, o))
        return graph
