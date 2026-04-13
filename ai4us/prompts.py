"""Registry of every prompt variant used across the paper.

The paper explores four main axes in its zero-shot generation experiments:

1. **Geographic anchor** (``country``, ``china``, ``china_2020``, ...)
2. **Real-world reference** (``with_real_reference`` vs not)
3. **Theory guidance** (``with_theory`` vs not)
4. **Response schema** (basic 4-column, extended 6-column with POIs, ...)

Every variant that was ever tried during the research — including the ones
that ended up commented out in the legacy scripts — is preserved here as an
addressable entry. Experiment scripts refer to prompts by slug (e.g.
``"scaling_law.baseline"``) rather than passing raw strings around.

The :func:`get` helper raises a useful error listing available slugs, which
makes typos obvious.
"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent


@dataclass(frozen=True)
class Prompt:
    slug: str
    description: str
    body: str

    def __str__(self) -> str:
        return self.body


# ---------------------------------------------------------------------------
# Scaling law (Experiment 01)
# ---------------------------------------------------------------------------

_SCALING_LAW: list[Prompt] = [
    Prompt(
        slug="scaling_law.baseline",
        description="Country-agnostic baseline. No real-world or theory reference.",
        body=dedent("""\
            Generate the dataset containing 100 cities in a country.
            Format the output as:
            CityName1, Population1, Infrastructure volume1, GDP1
            CityName2, Population2, Infrastructure volume2, GDP2
            ...
            Output exactly 100 lines without any additional text or explanations.
            - City Name
            - Population
            - Infrastructure volume: total road miles
            - GDP: all Gross Domestic Product of the city in one year
        """),
    ),
    Prompt(
        slug="scaling_law.china_2020",
        description="Anchored to China in 2020.",
        body=dedent("""\
            Generate the dataset containing 100 cities in china in 2020.
            Format the output as:
            CityName1, Population1, Infrastructure volume1, GDP1
            CityName2, Population2, Infrastructure volume2, GDP2
            ...
            Output exactly 100 lines without any additional text or explanations.
            - City Name
            - Population
            - Infrastructure volume: total road miles
            - GDP: all Gross Domestic Product of the city in one year
        """),
    ),
    Prompt(
        slug="scaling_law.china",
        description="Anchored to China without explicit year.",
        body=dedent("""\
            Generate the dataset containing 100 cities in china.
            Format the output as:
            CityName1, Population1, Infrastructure volume1, GDP1
            CityName2, Population2, Infrastructure volume2, GDP2
            ...
            Output exactly 100 lines without any additional text or explanations.
            - City Name
            - Population
            - Infrastructure volume: total road miles
            - GDP: all Gross Domestic Product of the city in one year
        """),
    ),
    Prompt(
        slug="scaling_law.with_real_reference",
        description="Baseline + 'refer to real-world city data'.",
        body=dedent("""\
            Generate the dataset containing 100 cities in a country. Please refer to the real world city data for data generation.
            Format the output as:
            CityName1, Population1, Infrastructure volume1, GDP1
            CityName2, Population2, Infrastructure volume2, GDP2
            ...
            Output exactly 100 lines without any additional text or explanations.
            - City Name
            - Population
            - Infrastructure volume: total road miles
            - GDP: all Gross Domestic Product of the city in one year
        """),
    ),
    Prompt(
        slug="scaling_law.with_theory",
        description="Baseline + 'under the guidance of urban scaling law and Zipf law'.",
        body=dedent("""\
            Generate the dataset containing 100 cities in a country. Please generate data under the guidance of urban scaling law and Zipf' law.
            Format the output as:
            CityName1, Population1, Infrastructure volume1, GDP1
            CityName2, Population2, Infrastructure volume2, GDP2
            ...
            Output exactly 100 lines without any additional text or explanations.
            - City Name
            - Population
            - Infrastructure volume: total road miles
            - GDP: all Gross Domestic Product of the city in one year
        """),
    ),
    Prompt(
        slug="scaling_law.with_real_and_theory",
        description="Combines real-world reference and theory guidance.",
        body=dedent("""\
            Generate the dataset containing 100 cities in a country. Please refer to the real world city data and under the guidance of urban scaling law and Zipf' law for data generation.
            Format the output as:
            CityName1, Population1, Infrastructure volume1, GDP1
            CityName2, Population2, Infrastructure volume2, GDP2
            ...
            Output exactly 100 lines without any additional text or explanations.
            - City Name
            - Population
            - Infrastructure volume: total road miles
            - GDP: all Gross Domestic Product of the city in one year
        """),
    ),
    Prompt(
        slug="scaling_law.detailed_schema",
        description="Extended schema with POIs + integer constraints.",
        body=dedent("""\
            Generate complete dataset containing 100 cities in a country. Don't have any other answers except generating data. Do not base generation on any theoretical assumptions. It is better to conform to the real data distribution of the real world. Don't remember the value generated by the previous city before the data of each city is generated, and ensure randomness. The attributes for each city are as follows:
            - City Name
            - Population
            - Infrastructure volume: Total floor area of buildings
            - Number of POIs: POIs of commercial facilities
            - GDP: all Gross Domestic Product of the city in one year

            The data for each city should meet the following criteria:
            - Population: Return an integer value.
            - Infrastructure volume: Return an integer value.
            - Number of POIs: Return an integer value.
            - GDP: Return an integer value.

            Ensure that the data format is as follows:
            City Name, Population, Infra, POIs, GDP
            And that all values are integers, without units or additional symbols.
        """),
    ),
    Prompt(
        slug="scaling_law.infra_as_floor_area",
        description="Alternative infrastructure definition: 'Total floor area of buildings' instead of road miles.",
        body=dedent("""\
            Generate the dataset containing 100 cities in a country.
            Format the output as:
            CityName1, Population1, Infrastructure volume1, GDP1
            CityName2, Population2, Infrastructure volume2, GDP2
            ...
            Output exactly 100 lines without any additional text or explanations.
            - City Name
            - Population
            - Infrastructure volume: Total floor area of buildings
            - GDP: all Gross Domestic Product of the city in one year
        """),
    ),
    Prompt(
        slug="scaling_law.claude_real_reference",
        description="Claude-script variant: 'Refer to the real world city data'.",
        body=dedent("""\
            Generate the dataset containing 100 cities in a country. Refer to the real world city data for data generation.
            Format the output as:
            CityName1, Population1, Infrastructure volume1, GDP1
            CityName2, Population2, Infrastructure volume2, GDP2
            ...
            Output exactly 100 lines without any additional text or explanations.
            - City Name
            - Population
            - Infrastructure volume: total road miles
            - GDP: all Gross Domestic Product of the city in one year
        """),
    ),
    Prompt(
        slug="scaling_law.claude_theory",
        description="Claude-script variant: 'Refer to urban scaling law and Zipf's law'.",
        body=dedent("""\
            Generate the dataset containing 100 cities in a country. Refer to urban scaling law and Zipf's law for data generation.
            Format the output as:
            CityName1, Population1, Infrastructure volume1, GDP1
            CityName2, Population2, Infrastructure volume2, GDP2
            ...
            Output exactly 100 lines without any additional text or explanations.
            - City Name
            - Population
            - Infrastructure volume: total road miles
            - GDP: all Gross Domestic Product of the city in one year
        """),
    ),
    Prompt(
        slug="scaling_law.claude_real_and_theory",
        description="Claude-script variant combining real-world and theory.",
        body=dedent("""\
            Generate the dataset containing 100 cities in a country. Refer to the urban scaling law and Zipf's law and under the guidance of real world city data for data generation.
            Format the output as:
            CityName1, Population1, Infrastructure volume1, GDP1
            CityName2, Population2, Infrastructure volume2, GDP2
            ...
            Output exactly 100 lines without any additional text or explanations.
            - City Name
            - Population
            - Infrastructure volume: total road miles
            - GDP: all Gross Domestic Product of the city in one year
        """),
    ),
]


# ---------------------------------------------------------------------------
# Distance decay (Experiment 02)
# ---------------------------------------------------------------------------

_DISTANCE_DECAY: list[Prompt] = [
    Prompt(
        slug="distance_decay.baseline_100",
        description="100 concentric rings, two attributes (pop density + land density).",
        body=dedent("""\
            Generate attributes for 100 concentric city circles arranged in order in a city.
            From the city center (Circle 1) to the outermost ring (Circle 100).
            Format the output as:
            Circle 1, Population Density1, Land Density 1
            Circle 2, Population Density2, Land Density 2
            ...
            Output exactly 100 lines without any additional text or explanations.
            - Population Density.
            - Land Density: Defined as the ratio of impervious surface area to the available land area.
        """),
    ),
    Prompt(
        slug="distance_decay.baseline_100_detailed_instructions",
        description="Same as baseline but with more explicit formatting instructions.",
        body=dedent("""\
            Generate data for 100 concentric city rings (circle layers) arranged in order
            from the city center (Circle 1) to the outermost ring (Circle 100).

            Each layer should have:
            1) A circle layer name or index (e.g., "Circle 1", "Circle 2", ..., "Circle 100").
            2) A population density value.
            3) A land density value. Defined as the ratio of impervious surface area to the available land area (excluding rivers, large lakes, protected areas, etc.).

            Instructions:
            - Simply generate 100 lines of data, one for each concentric ring from the city center to the outermost,
              in the format:
              Circle Layer, Population Density, Land Density
            - No units, no extra text or explanation, no headers, no footers.
            - Each line must be comma-separated with exactly three values.
            - The ordering goes from Circle 1 (closest to center) to Circle 100 (farthest).

            Generate exactly 100 lines corresponding to these 100 rings.
        """),
    ),
    Prompt(
        slug="distance_decay.city_150rings_2010",
        description="150 rings with 1 km spacing, anchored to a 2010 city with outer ring road.",
        body=dedent("""\
            Generate attributes for a city divided into 150 concentric rings (1 km each) in 2010, from the center (Circle 1) to the outermost area (Circle 150). The city boundary should be large enough to cover both the urban core and the outer ring road, forming an integrated urban system.
            Format the output as:
            Circle 1, Population Density1, Land Density 1
            Circle 2, Population Density2, Land Density 2
            ...
            Output exactly 150 lines without any additional text or explanations.
            Output only numerical values.
            - Population Density (people/km²).
            - Land Density: Defined as the ratio of impervious surface area to the available land area.
        """),
    ),
    Prompt(
        slug="distance_decay.with_real_reference",
        description="Baseline + 'refer to real world city data'.",
        body=dedent("""\
            Generate attributes for 100 concentric city circles arranged in order in a city. Refer to the real world city data for data generation.
            From the city center (Circle 1) to the outermost ring (Circle 100).
            Format the output as:
            Circle 1, Population Density1, Land Density 1
            Circle 2, Population Density2, Land Density 2
            ...
            Output exactly 100 lines without any additional text or explanations.
            - Population Density.
            - Land Density: Defined as the ratio of impervious surface area to the available land area.
        """),
    ),
    Prompt(
        slug="distance_decay.with_theory",
        description="Baseline + 'refer to distance decay law'.",
        body=dedent("""\
            Generate attributes for 100 concentric city circles arranged in order in a city. Refer to distance decay law for data generation.
            From the city center (Circle 1) to the outermost ring (Circle 100).
            Format the output as:
            Circle 1, Population Density1, Land Density 1
            Circle 2, Population Density2, Land Density 2
            ...
            Output exactly 100 lines without any additional text or explanations.
            - Population Density.
            - Land Density: Defined as the ratio of impervious surface area to the available land area.
        """),
    ),
    Prompt(
        slug="distance_decay.with_real_and_theory",
        description="Combined real-world anchor + theory anchor.",
        body=dedent("""\
            Generate attributes for 100 concentric city circles arranged in order in a city. Refer to the distance decay law and under the guidance of real world city data for data generation.
            From the city center (Circle 1) to the outermost ring (Circle 100).
            Format the output as:
            Circle 1, Population Density1, Land Density 1
            Circle 2, Population Density2, Land Density 2
            ...
            Output exactly 100 lines without any additional text or explanations.
            - Population Density.
            - Land Density: Defined as the ratio of impervious surface area to the available land area.
        """),
    ),
]


# ---------------------------------------------------------------------------
# Urban vitality — block attribute generation (Experiment 03, step 1)
# ---------------------------------------------------------------------------

_VITALITY_ATTRIBUTES: list[Prompt] = [
    Prompt(
        slug="vitality_attributes.jacobs_five",
        description="Jacobs' five-element schema: population density, building mix, short block, aged building, tall building.",
        body=dedent("""\
            Generate attributes for 100 blocks in a city.
            Format the output as:
            Block Name1, Population Density1, Building Mix Index1, Short Block1, Aged Building1, Tall Building1
            Block Name2, Population Density2, Building Mix Index2, Short Block2, Aged Building2, Tall Building2
            ...
            Output exactly 100 lines without any additional text or explanations.
            - Population Density: People per square meter.
            - Building Mix Index: Entropy index based on builidng use. Return value 0 and 1.
            - Short Block: Total number of street intersections divided by the area of the block.
            - Aged Building: The rate of old buildings on each block. Return value 0 and 1.
            - Tall Building: The average building heights on each block.
        """),
    ),
    Prompt(
        slug="vitality_attributes.street_density_scale_age",
        description="Alternative schema using street density, block scale, and building age instead of short/tall block.",
        body=dedent("""\
            Generate a dataset containing attributes for 100 blocks. Don't have any other answers except generating data. Do not base generation on any theoretical assumptions. It is better to conform to the real data distribution of the real world. Don't remember the value generated by the previous blocks before the data of each block is generated, and ensure randomness. The attributes for each block are as follows:
            - Block name.
            - Population Density: people per square meter.
            - Building Mix Index: entropy index based on builidng use.
            - Street Density: meters of street per square meter.
            - Block Scale: area of each block.
            - Building Age: the rate of old buildings on each block.

            The data for each block should meet the following criteria:
            - Population Density.
            - Building Mix Index: Return between 0 and 1.
            - Street Density: Return value 0 and 1.
            - Block Scale: Return an integer value.
            - Building Age: Return value 0 and 1.

            Ensure that the data format is as follows:
            Block Name, Population Density, Building Mix Index, Street Density, Block Scale, Building Age
            And that all values without any units or additional symbols.
        """),
    ),
]


# ---------------------------------------------------------------------------
# Urban vitality — livability scoring (Experiment 03, step 2)
# ---------------------------------------------------------------------------

_VITALITY_SCORING: list[Prompt] = [
    Prompt(
        slug="vitality_scoring.direct_json",
        description="Direct scoring, JSON output, no persona.",
        body=dedent("""\
            Assign a livability score between 0 and 1 for each block,
            where 0 represents the least livable and 1 represents the most livable.

            Evaluate each block based on the following attributes:
            - Block name.
            - Population Density: People per square meter.
            - Building Mix Index: Entropy index based on building use.
            - Short Block: Total number of street intersections divided by the area of the block.
            - Aged Building: The rate of old buildings on each block.
            - Tall Building: The average building heights on each block.

            **Instructions for the response:**
            1. Provide the scores in JSON format with block names as keys and scores as values.
            2. Do not include any additional text or explanations.
            3. Ensure the JSON is properly formatted for easy parsing.
            4. Do not use code blocks or any markdown formatting.
        """),
    ),
    Prompt(
        slug="vitality_scoring.as_resident",
        description="'You are a resident of a city' persona + JSON output.",
        body=dedent("""\
            You are a resident of a city.
            Assign a livability score between 0 and 1 for each block,
            where 0 represents the least livable and 1 represents the most livable.

            Evaluate each block based on the following attributes:
            - Block name.
            - Population Density: People per square meter.
            - Building Mix Index: Entropy index based on building use.
            - Short Block: Total number of street intersections divided by the area of the block.
            - Aged Building: The rate of old buildings on each block.
            - Tall Building: The average building heights on each block.

            **Instructions for the response:**
            1. Provide the scores in JSON format with block names as keys and scores as values.
            2. Do not include any additional text or explanations.
            3. Ensure the JSON is properly formatted for easy parsing.
            4. Do not use code blocks or any markdown formatting.
        """),
    ),
    Prompt(
        slug="vitality_scoring.jacobs_theory",
        description="Scoring anchored to Jane Jacobs' urban vitality theory.",
        body=dedent("""\
            Refer to Jane Jacobs' urban vitality theory for access vitality.
            Assign a livability score between 0 and 1 for each block,
            where 0 represents the least livable and 1 represents the most livable.

            Evaluate each block based on the following attributes:
            - Block name.
            - Population Density: People per square meter.
            - Building Mix Index: Entropy index based on building use.
            - Short Block: Total number of street intersections divided by the area of the block.
            - Aged Building: The rate of old buildings on each block.
            - Tall Building: The average building heights on each block.

            **Instructions for the response:**
            1. Provide the scores in JSON format with block names as keys and scores as values.
            2. Do not include any additional text or explanations.
            3. Ensure the JSON is properly formatted for easy parsing.
            4. Do not use code blocks or any markdown formatting.
        """),
    ),
    Prompt(
        slug="vitality_scoring.resident_plus_theory",
        description="Resident persona + Jacobs theory anchor.",
        body=dedent("""\
            You are a resident of a city. Refer to Jane Jacobs' urban vitality theory for access vitality.
            Assign a livability score between 0 and 1 for each block,
            where 0 represents the least livable and 1 represents the most livable.

            Evaluate each block based on the following attributes:
            - Block name.
            - Population Density: People per square meter.
            - Building Mix Index: Entropy index based on building use.
            - Short Block: Total number of street intersections divided by the area of the block.
            - Aged Building: The rate of old buildings on each block.
            - Tall Building: The average building heights on each block.

            **Instructions for the response:**
            1. Provide the scores in JSON format with block names as keys and scores as values.
            2. Do not include any additional text or explanations.
            3. Ensure the JSON is properly formatted for easy parsing.
            4. Do not use code blocks or any markdown formatting.
        """),
    ),
    Prompt(
        slug="vitality_scoring.resident_with_weights",
        description="Alternative schema: return livability score + per-attribute weights.",
        body=dedent("""\
            You are a resident of a city. Evaluate each block based on the following attributes:
            - Population Density: people per square meter.
            - Building Mix Index: entropy index based on building use.
            - Street Density: meters of street per square meter.
            - Block Scale: area of each block.
            - Building Age: average construction age of buildings on each block.

            Please assign a livability score between 0 and 1 for each block considering the attributes above, where 0 represents the least livable and 1 represents the most livable.

            **Instructions for the response:**
            1. Provide only the numerical values in the following format:
                Livability Score: <score>
                Population Density Weight: <weight>
                Building Mix Index Weight: <weight>
                Street Density Weight: <weight>
                Block Scale Weight: <weight>
                Building Age Weight: <weight>
            2. Do not include any additional text or explanations.
            3. Ensure each line follows the exact format for easy parsing.
            4. All values must be floating-point numbers between 0 and 1.
        """),
    ),
]


# ---------------------------------------------------------------------------
# Perception — pairwise judgement on Place Pulse 2.0 (Experiment 04)
# ---------------------------------------------------------------------------

_PERCEPTION_PAIRWISE: list[Prompt] = [
    Prompt(
        slug="perception.pairwise_six_dimensions",
        description="Pairwise perceptual comparison across the six Place Pulse dimensions.",
        body=dedent("""\
            You are shown two street-view photographs, labelled LEFT and RIGHT.

            For the perceptual dimension '{dimension}', decide which image rates higher.
            Reply with exactly one token: "LEFT", "RIGHT", or "EQUAL" (use "EQUAL" only if the two images are truly indistinguishable along this dimension).

            Dimension definitions:
            - safety: Which place looks safer?
            - beautiful: Which place looks more beautiful?
            - lively: Which place looks livelier?
            - wealthy: Which place looks wealthier?
            - depressing: Which place looks more depressing?
            - boring: Which place looks more boring?
        """),
    ),
]


# ---------------------------------------------------------------------------
# Synthetic intervention — image generation (Experiment 05)
# ---------------------------------------------------------------------------

_IMAGE_BASELINE: list[Prompt] = [
    Prompt(
        slug="image.baseline_single",
        description="Single generic prompt used to establish a minimal-diversity baseline.",
        body="A photorealistic street-view photograph of a typical urban scene.",
    ),
    Prompt(
        slug="image.structured_diverse",
        description="Structured prompt that combines culture/time/density/vegetation levers to maximize diversity.",
        body=dedent("""\
            A photorealistic street-view photograph taken at eye level. Scene attributes:
            - Region: {region}
            - Time of day: {time_of_day}
            - Density: {density}
            - Dominant land use: {land_use}
            - Vegetation level: {vegetation}
            - Road type: {road_type}
            Camera: 35mm lens, natural lighting, neutral color grading, no text overlays.
        """),
    ),
]

_IMAGE_INTERVENTION: list[Prompt] = [
    Prompt(
        slug="image.intervention_natural",
        description="Add natural elements (trees, grass, sky, water, bushes) to a baseline image.",
        body="Preserve the composition of the input image but add trees, grass, and natural vegetation. Keep all other elements intact.",
    ),
    Prompt(
        slug="image.intervention_traffic",
        description="Add traffic elements (cars, trucks, bridges, roads) to a baseline image.",
        body="Preserve the composition of the input image but add cars, trucks, and visible road infrastructure. Keep all other elements intact.",
    ),
    Prompt(
        slug="image.intervention_built",
        description="Add built elements (buildings, fences, walls) to a baseline image.",
        body="Preserve the composition of the input image but add buildings, walls, and fences in a natural urban configuration. Keep all other elements intact.",
    ),
]


# ---------------------------------------------------------------------------
# SynAlign pipeline — perceptual stage 1 (Experiment 08)
# ---------------------------------------------------------------------------

_SYNALIGN_PERCEPTUAL: list[Prompt] = [
    Prompt(
        slug="synalign.perceptual.stage1_discovery",
        description="Discovery stage: synthesise a Perceptual Mapping Blueprint across six Place Pulse dimensions.",
        body=dedent("""\
            [Task] Synthesize a "Perceptual Mapping Blueprint" through a three-step empirical discovery process.
            You MUST provide a separate, dedicated analysis for EACH of the six dimensions: Safety, Beautiful, Lively, Wealthy, Boring, and Depressing.

            ### Step 1: Dimension-Specific Qualitative Archetypes (Visual Drivers)
            For EACH dimension separately:
            - Define exactly what 'Typical High', 'Typical Low', and 'Ambiguous' look like.
            - Identify the unique "Visual Switches": Which specific urban elements (e.g., street furniture, facade texture, vegetation density) drive THIS specific perception?

            ### Step 2: Dimension-Specific Quantitative Scaling (CLIP Signal Mapping)
            Analyze the "LARGE-SCALE NUMERICAL MAPPING DATA" ({N_QUANTITATIVE_ANCHORS} samples per category) provided:
            - For EACH dimension, establish the statistical correlation between the 8-dimensional CLIP signals and Human Scores ($\\mu$).
            - Identify the "Semantic Signatures": Which specific CLIP dimensions act as the primary "Perceptual Thermometers" for this quality?
              (e.g., "In 'Safety', Positive Dim 2 + Negative Dim 5 = High mu; while in 'Wealthy', Dim 1 is the dominant driver.")

            ### Step 3: Precise Boundary Logic & The 'Equal' Threshold
            For EACH dimension, define the "Zone of Indifference":
            - Analyze the 'Ambiguous' cases to identify the visual and numerical "Dead-Zones" where differences are too contradictory or subtle to pick a winner.
            - FORMULATE THE 'EQUAL' RULE: Provide explicit thresholds.
              (e.g., "Choose 'Equal' for 'Beautiful' if the CLIP signal delta between images is < 0.15 AND both images lack primary visual drivers identified in Step 1.")

            [Output Requirement]
            Output a structured scientific manual in English. Organize the content category by category to ensure NO dimension is generalized or skipped. Focus on the explicit mapping between CLIP signals and visual cues.
        """),
    ),
]


# ---------------------------------------------------------------------------
# Flat registry + lookup
# ---------------------------------------------------------------------------

_ALL: list[Prompt] = (
    _SCALING_LAW
    + _DISTANCE_DECAY
    + _VITALITY_ATTRIBUTES
    + _VITALITY_SCORING
    + _PERCEPTION_PAIRWISE
    + _IMAGE_BASELINE
    + _IMAGE_INTERVENTION
    + _SYNALIGN_PERCEPTUAL
)

REGISTRY: dict[str, Prompt] = {p.slug: p for p in _ALL}


def get(slug: str) -> Prompt:
    """Return the :class:`Prompt` registered under ``slug``.

    Raises :class:`KeyError` with a useful message listing all available slugs
    (filtered by the slug's category prefix) when the lookup fails.
    """
    if slug in REGISTRY:
        return REGISTRY[slug]
    category = slug.split(".", 1)[0]
    matches = sorted(s for s in REGISTRY if s.startswith(category + "."))
    raise KeyError(
        f"Unknown prompt slug {slug!r}. Available slugs in this category:\n  "
        + "\n  ".join(matches or sorted(REGISTRY))
    )


def list_slugs(category: str | None = None) -> list[str]:
    """List all registered slugs, optionally filtering by category prefix."""
    if category is None:
        return sorted(REGISTRY)
    return sorted(s for s in REGISTRY if s.startswith(category + "."))
