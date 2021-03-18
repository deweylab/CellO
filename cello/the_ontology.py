from .onto_lib_py3 import load_ontology

GO_ONT_CONFIG_ID = '18'
UNIT_OG_ID = '7'

ONTO_PATCH = {
    "add_edges": [
        {
            "source_term": "CL:2000001",    # peripheral blood mononuclear cell
            "target_term": "CL:0000081",    # blood cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000670",    # primordial germ cell
            "target_term": "CL:0002321",    # embryonic cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0008001",    # hematopoietic precursor cell
            "target_term": "CL:0011115",    # precursor cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0002246",    # peripheral blood stem cell
            "target_term": "CL:0000081",    # blood cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000542",    # lymphocyte
            "target_term": "CL:0000842",    # mononuclear cell
            "edge_type": "is_a" 
        },
        {
            "source_term": "CL:0000066",    # epithelial cell
            "target_term": "CL:0002371",    # somatic cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0001035",    # bone cell
            "target_term": "CL:0002371",    # somatic cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000018",    # spermatid
            "target_term": "CL:0011115",    # precursor cell  
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000017",    # spermatocyte
            "target_term": "CL:0011115",    # precursor cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000235",    # macrophage
            "target_term": "CL:0000842",    # mononuclear cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000235",    # macrophage
            "target_term": "CL:0000145",    # professional antigen presenting cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000451",    # dendritic cell
            "target_term": "CL:0000145",    # professional antigen presenting cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000236",    # B cell
            "target_term": "CL:0000145",    # professional antigen presenting cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0002371",    # somatic cell
            "target_term": "CL:0000255",    # eukaryotic cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000163",    # endocrine cell
            "target_term": "CL:0002371",    # somatic cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0008024",    # pancreatic endocrine cell
            "target_term": "CL:0000164",    # enteroendocrine cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000583",    # alveolar macrophage
            "target_term": "CL:1001603",    # lung macrophage
            "edge_type": "is_a"
        },
        {
            'source_term': 'CL:0000091',        # Kupffer cell
            "target_term": "UBERON:0002107",    # liver
            "edge_type": "part_of"
        }
    ]
}


def patch_the_ontology(og):
    for edge_info in ONTO_PATCH['add_edges']:
        source_id = edge_info['source_term']
        target_id = edge_info['target_term']
        source_term = None
        target_term = None
        if source_id in og.id_to_term:
            source_term = og.id_to_term[source_id]
        if target_id in og.id_to_term:
            target_term = og.id_to_term[target_id]
        if source_term is None or target_term is None:
            continue
        edge_type = edge_info['edge_type']
        inv_edge_type = "inv_%s" % edge_type
        if edge_type in source_term.relationships:
            source_term.relationships[edge_type].append(target_id)
        else:
            source_term.relationships[edge_type] = [target_id]
        if inv_edge_type in target_term.relationships:
            target_term.relationships[inv_edge_type].append(source_id)
        else:
            target_term.relationships[inv_edge_type] = [source_id]
    return og

#ONT_NAME_TO_ONT_ID = {"EFO_CL_DOID_UBERON_CVCL":"17"}

#ont_id_to_og = None
#def _ont_id_to_og():
#    global ont_id_to_og
#    if ont_id_to_og is None:
#        ont_id_to_og = {x: patch_the_ontology(load_ontology.load(x)[0]) for x in ONT_NAME_TO_ONT_ID.values()}
#    return ont_id_to_og

def the_ontology():
    return patch_the_ontology(load_ontology.load('17')[0])
    #return _ont_id_to_og()['17']

def unit_ontology():
    return load_ontology.load(UNIT_OG_ID)[0]

def go_ontology():
    return load_ontology.load(GO_ONT_CONFIG_ID)[0] 


def main():
    #og = the_ontology()
    #og = patch_the_ontology(og)
    #print og.id_to_term['CL:0000081']
    #print og.id_to_term['CL:2000001']
    #print og.id_to_term['CL:0000542']

    og = go_ontology()
    print(og.id_to_term['GO:0002312'])

if __name__ == "__main__":
    main()
