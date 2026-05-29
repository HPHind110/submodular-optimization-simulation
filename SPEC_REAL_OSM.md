\# SPEC\_REAL\_OSM: Real-world OSM experiments for submodular optimization



\## Goal



Upgrade the current synthetic simulation project into a real-world geospatial experiment using OpenStreetMap data.



The main application is:



Choosing public Wi-Fi station locations in an urban area so that many demand points are covered or well served.



The experiment must remain aligned with the theory in the report:



\- Maximum Coverage

\- Facility Location

\- Greedy

\- Lazy Greedy

\- Stochastic Greedy



Do not implement deep learning or continual learning.



\## Study area



Default area:



"Hoan Kiem, Hanoi, Vietnam"



Allow overriding this area from command line arguments.



\## Data source



Use OpenStreetMap through OSMnx.



Demand points should be collected from OSM features such as:

\- amenity: school, hospital, clinic, university, library, restaurant, cafe

\- tourism: attraction, museum

\- public\_transport: platform

\- highway: bus\_stop



Candidate facility locations should be:

\- bus stops

\- public transport platforms

\- optionally sampled road network nodes



If OSM data is too sparse for one category, combine categories.



\## Coordinate system



Downloaded OSM geometries are usually in latitude/longitude. Convert them to a projected CRS before computing distances in meters.



Use GeoPandas / OSMnx projection utilities.



\## Processed data



Save processed data to:

\- data/processed/demand\_points.csv

\- data/processed/candidate\_points.csv



Each CSV must contain:

\- id

\- x

\- y

\- lon

\- lat

\- source\_type



\## Maximum Coverage formulation



Given demand points U and candidate points P:



A candidate p\_j covers demand point u\_i if distance(u\_i, p\_j) <= R.



Default:

R = 300 meters

k = 10



For selected candidates I:

f(I) = number of covered demand points



Metrics:

\- objective\_value

\- coverage\_count

\- coverage\_rate

\- average\_nearest\_distance\_m

\- max\_nearest\_distance\_m

\- eval\_count

\- runtime\_seconds



Algorithms:

\- Greedy

\- Lazy Greedy

\- Stochastic Greedy

\- Random Baseline



Do not use brute force for real OSM data unless data is very small.



\## Facility Location formulation



Use the same demand points and candidate points.



Similarity:

w\_ij = exp(-d(u\_i, p\_j)^2 / (2 sigma^2))



Default:

sigma = 300 meters

k = 10



Objective:

f(S) = sum\_i max\_{j in S} w\_ij



Metrics:

\- objective\_value

\- average\_nearest\_distance\_m

\- max\_nearest\_distance\_m

\- eval\_count

\- runtime\_seconds



Algorithms:

\- Greedy

\- Lazy Greedy

\- Stochastic Greedy

\- Random Baseline



\## Figures



Generate:

\- outputs/figures/osm\_points\_map.png

&#x20; Demand points and candidate locations.



\- outputs/figures/real\_coverage\_result.png

&#x20; Selected facilities, covered demand points, uncovered demand points, coverage radius.



\- outputs/figures/real\_facility\_result.png

&#x20; Selected representatives/facilities and assignment lines or nearest assignments.



\- outputs/figures/real\_runtime\_comparison.png

&#x20; Runtime comparison across algorithms.



\- outputs/figures/real\_evaluation\_comparison.png

&#x20; Evaluation count comparison across algorithms.



\## Tables



Generate:

\- outputs/tables/real\_coverage\_results.csv

\- outputs/tables/real\_coverage\_results.tex

\- outputs/tables/real\_facility\_results.csv

\- outputs/tables/real\_facility\_results.tex

\- outputs/tables/real\_algorithm\_comparison.csv

\- outputs/tables/real\_algorithm\_comparison.tex



\## Code structure



Add these files:



src/osm\_data.py

\- collect\_osm\_points(place\_name)

\- preprocess\_points(...)

\- save\_processed\_data(...)



src/geo\_metrics.py

\- pairwise\_distance\_matrix(...)

\- average\_nearest\_distance(...)

\- max\_nearest\_distance(...)



src/geo\_coverage.py

\- build\_coverage\_sets(...)

\- coverage\_objective\_geo(...)

\- coverage\_marginal\_gain\_geo(...)



src/geo\_facility\_location.py

\- build\_similarity\_matrix(...)

\- facility\_objective\_geo(...)

\- facility\_marginal\_gain\_geo(...)



experiments/run\_osm\_data\_collection.py

experiments/run\_real\_coverage.py

experiments/run\_real\_facility.py

experiments/run\_real\_algorithm\_comparison.py



\## CLI arguments



Scripts should support basic command line arguments:

\--place

\--k

\--radius

\--sigma

\--seed



\## Reliability



If OSMnx download fails or returns too few points:

\- print a clear error message

\- do not crash with obscure stack traces

\- allow using cached processed CSV files if they exist



\## Report style



Generate outputs that can be directly used in Chapter 4.

Do not write long theory in generated LaTeX tables.

