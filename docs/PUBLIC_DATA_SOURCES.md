# Public data source catalog

This catalog is a starting point for project discovery, not a promise that
every record is unrestricted or project-ready. Before downloading anything,
read the dataset-level license, terms, citation request, access controls, data
dictionary, provenance, update policy, and human-subject restrictions.

Prefer a source with stable identifiers, machine-readable metadata, a clear
observation unit, documented collection methods, and a reproducible API or
download URL. Record the exact dataset, version or retrieval date, query, and
checksum in `docs/DATASET_CARD.md`.

## General-purpose and machine-learning catalogs

| Source | Areas and access | Good starting questions |
| --- | --- | --- |
| [UCI Machine Learning Repository](https://archive.ics.uci.edu/) | Curated tabular, time-series and multivariate datasets; per-dataset licenses | Compare course algorithms on a bounded, documented task |
| [OpenML](https://www.openml.org/search?type=data) | Versioned ML datasets, tasks and benchmark metadata; API and Python client | Reproduce a benchmark while auditing its assumptions |
| [Data.gov](https://catalog.data.gov/dataset/) | U.S. federal and participating local catalogs; files and catalog API | Join public measurements with geographic or policy context |
| [World Bank Data Catalog](https://datacatalog.worldbank.org/) | International development, economy, population and environment | Model trends while respecting country and time dependence |
| [United Nations Data](https://data.un.org/) | International demographic, social, economic and environmental indicators | Compare trajectories without treating countries as IID rows |
| [European Data Portal](https://data.europa.eu/en) | European Union and member-state open-data catalog | Study transport, environment, government or regional systems |

## Climate, Earth and environmental systems

| Source | Areas and access | Engineering cautions |
| --- | --- | --- |
| [NOAA Climate Data Online](https://www.ncei.noaa.gov/cdo-web/) | Weather stations, temperature, precipitation, wind, normals; web services | Stations move, sensors and coverage change, and time splits matter |
| [NASA Earthdata Search](https://search.earthdata.nasa.gov/) | Satellite and Earth-system observations; spatial/temporal search and APIs | Files may be large, gridded and remote; prototype with bounded regions |
| [USGS Water Data for the Nation](https://waterdata.usgs.gov/) | Streamflow, gage height, groundwater and water quality; REST APIs | Measurements are site- and time-dependent with irregular missingness |
| [EPA AirData](https://www.epa.gov/outdoor-air-quality-data) | U.S. monitor-level air-quality measurements and AQI data; files and API | Monitor placement and reporting frequency affect apparent exposure |
| [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) | Global observations, reanalysis and projections; API account | Distinguish observations, model output and reanalysis products |
| [USGS EarthExplorer](https://earthexplorer.usgs.gov/) | Landsat, elevation, aerial imagery and other geospatial products | Coordinate reference systems, clouds and spatial leakage require care |
| [NOAA Fisheries Data](https://www.fisheries.noaa.gov/resources/data-and-maps) | Fisheries, ecosystems, species and ocean observations | Sampling effort and regulatory changes can confound trends |

## Biology, ecology and agriculture

| Source | Areas and access | Engineering cautions |
| --- | --- | --- |
| [GBIF](https://www.gbif.org/occurrence/search) | Global species-occurrence records; downloads and API | Presence records are not absence records; collection effort is uneven |
| [NSF NEON Data Portal](https://www.neonscience.org/data) | Open ecosystem, organismal, atmospheric and remote-sensing products | Sites, instruments, units and quality flags must remain explicit |
| [Environmental Data Initiative](https://portal.edirepository.org/nis/home.jsp) | Ecological and environmental research packages with metadata | Schemas vary; preserve package identifiers and methods |
| [USDA NASS](https://www.nass.usda.gov/datasets/) | Crops, livestock, prices, demographics and Census of Agriculture; API/files | Survey estimates have geographic, temporal and uncertainty structure |
| [USDA FoodData Central](https://fdc.nal.usda.gov/download-datasets.html) | Nutrient and food-composition records; downloads and API | Product versions, serving definitions and missing nutrients vary |

## Health, medicine and life science

Health data can remain sensitive even after direct identifiers are removed.
Use open aggregate or explicitly public data unless the course and institution
have approved the access, storage and analysis plan.

| Source | Areas and access | Engineering cautions |
| --- | --- | --- |
| [CDC Data](https://data.cdc.gov/) | Public-health surveillance and aggregate datasets; Socrata API | Surveillance definitions and reporting completeness change over time |
| [WHO Data](https://data.who.int/) | International health indicators and estimates | Countries differ in measurement systems and missingness |
| [CMS Data](https://data.cms.gov/) | U.S. Medicare/Medicaid providers, utilization and quality; files and API | Aggregate records do not justify individual clinical decisions |
| [ClinicalTrials.gov API](https://clinicaltrials.gov/data-api/about-api) | Study registrations, conditions, interventions and outcomes metadata | Registration records are not patient-level treatment evidence |
| [NCBI Gene Expression Omnibus](https://www.ncbi.nlm.nih.gov/geo/) | Public functional-genomics experiments and processed profiles | Batch, platform and study effects dominate naive row-level splits |
| [PubChem](https://pubchem.ncbi.nlm.nih.gov/docs/) | Chemical structures, properties, bioassays and programmatic APIs | Records integrate many sources; assay context and units are essential |
| [PhysioNet](https://physionet.org/about/database/) | Physiological signals and clinical datasets with mixed access levels | Some datasets require credentials, training or data-use agreements |
| [dbGaP](https://dbgap.ncbi.nlm.nih.gov/home/) | Genotype/phenotype studies with public metadata and controlled data | Individual-level human data usually requires institutional authorization |

## Physics, astronomy, chemistry and materials

| Source | Areas and access | Engineering cautions |
| --- | --- | --- |
| [CERN Open Data](https://opendata.cern/) | Particle-physics collision data, simulation, software and learning resources | Research-scale files can be enormous; begin with educational subsets |
| [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) | Confirmed planets, stellar hosts, survey and transit data; table API | Catalog selection effects and uncertainty shape every inference |
| [NASA Planetary Data System](https://pds.nasa.gov/) | Long-term archive for planetary missions and laboratory data | Mission products use specialized formats and calibration pipelines |
| [Sloan Digital Sky Survey](https://www.sdss.org/dr18/data_access/) | Imaging and spectra with SQL and bulk access | Cross-matching, detection limits and sky geometry affect independence |
| [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/) | Thermochemistry, spectra and physical properties | Verify units, phase, temperature and NIST reuse notes |
| [NIST Materials Data Repository](https://materialsdata.nist.gov/) | Experimental and computational materials records | Quality and terms are record-specific, not uniform across the portal |
| [Materials Project](https://next-gen.materialsproject.org/) | Computed materials properties and structures; API key | Predictions are computational results, not experimental ground truth |

## Economics, business and society

| Source | Areas and access | Engineering cautions |
| --- | --- | --- |
| [U.S. Census APIs](https://www.census.gov/data/developers/data-sets.html) | Population, housing, business, geography and survey estimates | Know the geography, universe, margin of error and vintage |
| [Bureau of Labor Statistics Public Data](https://www.bls.gov/data/) | Employment, prices, productivity, injuries and occupations; API/files | Series are revised, seasonally adjusted and frequency-dependent |
| [Bureau of Economic Analysis](https://www.bea.gov/data) | GDP, income, trade, industry and regional accounts; API | Releases and revisions make retrieval dates part of provenance |
| [FRED](https://fred.stlouisfed.org/) | Economic and financial time series from many publishers; API | FRED republishes series; preserve the original source and vintage |
| [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | Company submissions, filings and XBRL facts | Respect SEC fair-access limits and accounting-definition changes |
| [IRS Tax Statistics](https://www.irs.gov/statistics) | Aggregate tax, migration, business and nonprofit statistics | Aggregates and disclosure protection constrain interpretation |
| [IPUMS](https://www.ipums.org/) | Harmonized census and survey microdata with registration and terms | Extracts may require approval; weights and survey design are mandatory |

## Engineering, energy and transportation

| Source | Areas and access | Engineering cautions |
| --- | --- | --- |
| [U.S. EIA Open Data](https://www.eia.gov/opendata/) | Electricity, fuels, prices, generation and emissions; free API key | Time zones, units, revisions and operational hierarchies matter |
| [DOE Open Energy Data Initiative](https://data.openei.org/) | Energy-system, building, grid and renewable-resource datasets | Files range from small tables to cloud-scale simulation output |
| [NREL Data Catalog](https://data.nrel.gov/) | Renewable energy, buildings, mobility and grid research | Read experiment and simulation metadata before modeling |
| [Bureau of Transportation Statistics](https://www.bts.gov/browse-statistical-products-and-data) | Aviation, freight, passenger travel and infrastructure | Weather, route, carrier and calendar dependence complicate splits |
| [U.S. DOT Data Portal](https://data.transportation.gov/) | Road, vehicle, aviation and safety datasets; Socrata API | Reporting systems and jurisdiction coverage vary |
| [FAA Airport Data](https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger) | Passenger boarding and cargo statistics by airport | Airport roles and reporting definitions change over time |

## Geospatial, cities and public infrastructure

| Source | Areas and access | Engineering cautions |
| --- | --- | --- |
| [OpenStreetMap](https://www.openstreetmap.org/) | Global volunteered roads, places and infrastructure; APIs/extracts | ODbL obligations, completeness and tagging vary by region |
| [Natural Earth](https://www.naturalearthdata.com/downloads/) | Public-domain global vector and raster map layers | Appropriate for small-scale maps, not precise local boundaries |
| [NYC Open Data](https://opendata.cityofnewyork.us/) | City services, transport, buildings, environment and inspections | Complaint and enforcement data reflect reporting and policy processes |
| [Chicago Data Portal](https://data.cityofchicago.org/) | Public safety, transport, buildings, health and administration | Protect people when combining location and event data |
| [Houston Open Data](https://www.houstontx.gov/opendata/) | Local government, permitting, public works and geospatial resources | Confirm update schedules and department-specific definitions |

## Language, knowledge graphs and cultural records

| Source | Areas and access | Engineering cautions |
| --- | --- | --- |
| [Wikidata](https://www.wikidata.org/wiki/Wikidata:Data_access) | Multilingual structured knowledge graph; SPARQL, dumps and APIs | Community-edited completeness and ontology choices affect labels |
| [Library of Congress Data](https://data.labs.loc.gov/) | Digitized collections, metadata and APIs | Copyright and access rules are item- and collection-specific |
| [Project Gutenberg](https://www.gutenberg.org/policy/permission.html) | Public-domain and licensed ebooks in the United States | Copyright status can differ outside the United States |
| [GDELT](https://www.gdeltproject.org/data.html) | Global news-event and media-derived data streams | It measures media records and automated coding, not events directly |

## Choosing a feasible course dataset

Before committing to a source, answer:

- Can every teammate lawfully access it for the entire semester?
- Can CI use a small public or constructed fixture without a secret?
- Is the observation unit meaningful and is dependence identifiable?
- Are features available at the time the product would use them?
- Does the source document units, quality flags, missingness and revisions?
- Can a first end-to-end slice run on an ordinary laptop in minutes?
- Is there a defensible supervised or unsupervised question—not merely a file?
- Can the team explain harms from misuse, representation gaps and false claims?

Large or remote data are welcome when the project uses bounded queries,
streaming, chunking, databases or object storage. Do not commit a massive copy
to Git merely because the upstream source is public.
