from . import the_ontology

_cell_ontology = None
def cell_ontology():
    global _cell_ontology
    if _cell_ontology is None:
        _cell_ontology = the_ontology.the_ontology()
    return _cell_ontology

def get_term_name(term_id):
    try:
        return cell_ontology().id_to_term[term_id].name
    except KeyError:
        return None

