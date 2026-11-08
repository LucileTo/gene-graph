"""Shared definition of the knowledge graph: node types, closed list of mechanisms, relations, ALS node."""

# Node types
MECHANISM, DISEASE, MOLECULE = "Mechanism", "Disease", "Molecule"

# Closed list of mechanisms: id -> (name, what it covers, given to the LLM to map synonyms)
MECHANISMS = {
    "MECH:autophagy": (
        "autophagy",
        "autophagy, macroautophagy, mitophagy, autophagic flux, autophagosome formation, "
        "autophagy-lysosome pathway, clearance of protein aggregates by autophagy"),
    "MECH:oxidative_stress": (
        "oxidative stress",
        "oxidative stress, reactive oxygen species (ROS), free radicals, oxidative damage, "
        "lipid peroxidation, antioxidant defenses (Nrf2, glutathione, SOD activity)"),
    "MECH:excitotoxicity": (
        "excitotoxicity",
        "excitotoxicity, glutamate toxicity, overactivation of NMDA/AMPA/kainate receptors, "
        "glutamate-induced calcium overload"),
    "MECH:neuroinflammation": (
        "neuroinflammation",
        "neuroinflammation, microglial or astrocyte activation, pro-inflammatory cytokines in the "
        "nervous system (TNF-alpha, IL-1beta, IL-6), NLRP3 inflammasome in the brain"),
    "MECH:axonal_transport": (
        "axonal transport",
        "axonal transport, anterograde or retrograde transport, kinesin/dynein motors, "
        "microtubule-based transport along axons"),
}

# Relations: name -> (subject type, object type)
RELATIONS = {
    "involved_in": (MECHANISM, DISEASE),   # the mechanism takes part in the disease
    "inhibits": (MOLECULE, MECHANISM),
    "activates": (MOLECULE, MECHANISM),
    "induces": (MOLECULE, MECHANISM),
    "agonist_of": (MOLECULE, MECHANISM),
    "tested": (MOLECULE, DISEASE),         # pre-clinical or clinical study
    "treatment": (MOLECULE, DISEASE),      # established treatment
}

# ALS is excluded from the PubMed query: its node is added by hand at the very end
ALS_ID = "MESH:D000690"
ALS_NAME = "Amyotrophic Lateral Sclerosis"
ALS_RELATION = "documented_in_literature"  # Mechanism -> ALS, source = "manual"
