from . import the_ontology

CELL_ONTOLOGY = the_ontology.the_ontology()

def get_term_name(term_id):
    try:
        return CELL_ONTOLOGY.id_to_term[term_id].name
    except KeyError:
        return None

