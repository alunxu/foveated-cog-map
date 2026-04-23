# Narrative Revision Plan
## From deep-research agent, 2026-04-23

Research on foveated vision × cognitive maps and Wijmans 2023 narrative analysis.

---

## High-leverage moves (apply first)

### 1. Opening paragraph rewrite
Replace current §1¶1 ("Wijmans... Dwivedi... leaving open") with biology-first framing:

> Decades of animal-navigation research rest on a specific observation: what an animal foveates determines what its hippocampus encodes. Wirth et al. (2017) showed that primate hippocampal cells represent not just location but a joint code of place, view, and task context; Rolls (2024) argued that the primate fovea is precisely why primates have spatial-view cells while rats, with their near-panoramic retinae, have place cells. Yet when map-like representations have been shown to emerge in the recurrent memories of artificial navigation agents (Wijmans et al., 2023), it has always been under conditions of blind or spatially uniform perception — conditions no visual animal experiences. We ask the question that biological comparative studies cannot cleanly answer: when sensor structure is the only free variable, does the format of learned spatial memory change with it?

### 2. H1 biological anchor
Kupers & Ptito 2014 / Cattaneo 2008 (blind-human compensatory memory) + Chen et al. 2016 (grid cells in darkness).
Frame H1 finding as "learned-agent analog" of compensatory memory in congenitally blind humans.

### 3. H2 biological anchor
Geva-Sagiv et al. 2016 (bat hippocampus remaps between vision/echolocation) + Rolls 2024 (primate-vs-rat view/place cells).
"Our CKA<0.004 across all pairs... is the in-silico analog of these cross-modality and cross-species divergences."

### 4. H3 biological anchor
Jutras, Fries & Buffalo 2013 (saccades reset hippocampal theta, memory-encoding events) + Tas, Luck & Hollingworth 2016 (overt saccades write VWM, covert does not).
Each fixation = memory-encoding event; collapsed gaze = restricted encoding input.

### 5. Wijmans narrative moves to adopt
- Open on animal behaviour, not machines
- Multidisciplinary citations on page 1 (each maps agent → animal capability)
- Reframe architecture as reductive experimental preparation (vs "ablation")
- Stage findings as existence proofs / ontogenetic accounts
- Map each experiment onto a classical animal study
- Anthropomorphic-but-disciplined verbs with scare quotes
- "Convergent evolution" explanation pattern in Discussion
- Disclaim ML novelty, claim science novelty

### 6. Terminology shifts
- "hidden state" → "recurrent memory" / "agent's memory" in narrative prose (keep "hidden state" in Methods)
- "path-history probe" → "retrospective spatial memory probe" in Intro/Discussion
- "spatial non-uniformity" → "eccentricity-dependent visual acuity" in prose
- "cognitive map" only with Epstein 2017 operational definition
- "representational geometry" / "representational format" (Bellmund, Behrens, Kriegeskorte)
- "gaze-locked representational content" in Abstract/Intro; keep "dissociation" in Results

### 7. Avoid over-claims
- Don't say agents ARE cognitive maps → "exhibit signatures of" / "instantiate a form of"
- Don't say foveated has "place cells" → "units with high spatial information" / "Banino-et-al-style >1-bit units"
- Don't say fov-learned IS a head-direction system → "functional resemblance to head-direction coding"
- Keep "consistent with" hedge; never "identical to"
- H3 single-seed: keep "preliminary observation"

---

## Key references to add to literature.bib

### A. Foveation as information constraint
- Strasburger, Rentschler, Jüttner 2011 J Vis — eccentricity acuity falloff
- Rosenholtz 2016 Annu Rev Vis Sci — peripheral vision capabilities
- Land & Hayhoe 2001 Vis Res — eye movements in everyday activity
- Hayhoe 2017 Vis Res — gaze in natural behaviour
- Hayhoe & Matthis 2018 Phil Trans B — gaze in natural environments

### B. Hippocampal/entorhinal
- Wirth et al. 2017 PLoS Biol — primate hippocampal view-dependent coding [already cited]
- Rolls 2023 Hippocampus — spatial view cells review
- Rolls 2024 TiCS — primate fovea → view cells (vs rat place cells)
- Chen, King, Burgess, O'Keefe 2016 Curr Biol — grid cells collapse in darkness
- Jutras, Fries, Buffalo 2013 PNAS — saccades reset hippocampal theta, predict memory
- Hoffman et al. 2013 Front Syst Neurosci — saccade-aligned theta rhythm

### C. Cognitive maps
- Tolman 1948 Psych Rev — original "cognitive map"
- Epstein, Patai, Julian, Spiers 2017 Nat Neurosci — operational definition
- Behrens et al. 2018 Neuron — "What Is a Cognitive Map?"
- Bellmund et al. 2018 Science — spatial codes for human thinking
- Peer et al. 2021 TiCS — cognitive maps vs graphs

### D. Active vision × memory
- Castelhano & Henderson 2005 Vis Cogn / 2008 Psych Sci — fixations predict memorability
- Hollingworth & Henderson 2002 JEP:HPP — scene memory across saccades
- Henderson 2017 TiCS — gaze control as prediction
- Tas, Luck, Hollingworth 2016 JEP:HPP — overt vs covert encoding
- Tatler 2007 J Vis — central fixation bias
- Tatler et al. 2011 J Vis — reinterpreting salience

### E. Sensory-specific memory
- Yartsev & Ulanovsky 2013 Science — bat 3D place cells
- Geva-Sagiv et al. 2016 Nat Neurosci — bat hippocampal remapping across modalities
- Kupers, Ptito 2014 Neurosci Biobehav Rev — blind hippocampal reorganisation
- Cattaneo et al. 2008 — imagery and spatial processes in blindness
- Thaler, Arnott, Goodale 2011 PLoS ONE — blind echolocators recruit occipital cortex

### F. Cross-species divergence (mostly reuse B, E)
- Courellis et al. 2019 eLife — marmoset hippocampal theta
- Harten et al. 2020 Science — bat cognitive-map ontogeny

### G. Gaze as decision variable
- Najemnik & Geisler 2005 Nature — optimal eye movements for search
- Sprague & Ballard 2003 NeurIPS / 2007 Neural Comp — embodied visual behaviours RL
- Gottlieb, Oudeyer, Lopes, Baranes 2013 TiCS — information-seeking, curiosity, attention

---

## Structural moves

- Re-order §7 Related Work: neuroscience first, ML second
- Rename §5 Discussion → "Discussion: From foveated sensors to the format of spatial memory"
- Consider title v2: "Sensor Structure and Gaze Policy Shape the Representational Format of Spatial Memory in Navigation Agents"
