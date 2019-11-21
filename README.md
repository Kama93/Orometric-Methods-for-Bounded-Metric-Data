This repository contains the code and data for the paper **Orometric Methods
in Bounded Metric Data**.

# Data

The data sets for Germany and France were extracted via the Wikidata Query Service on 2019-11-04.
The following queries were used to extract the municipalities and the university
locations:

## Germany

### Municipalities

SELECT DISTINCT ?city ?cityLabel (MAX(?lat) AS ?latitude) (MAX(?long) AS ?longitude) (MAX(?pop) AS ?population) WHERE { 
  ?city (p:P31/ps:P31)/wdt:P279* wd:Q15284. 
  ?city wdt:P17 wd:Q183. 
  ?city wdt:P1082 ?pop. 
  filter (?pop >= 5000). 
  ?city (p:P625/psv:P625) ?coordinates. 
  ?coordinates wikibase:geoLatitude ?lat. 
  ?coordinates wikibase:geoLongitude ?long. 
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en, de". } 
} 
GROUP BY ?city ?cityLabel
ORDER BY DESC(?population)

### Universities:

SELECT DISTINCT ?city ?uni ?cityLabel ?uniLabel WHERE { 
  ?uni (p:P31/ps:P31)/wdt:P279* wd:Q3918. 
  ?uni (wdt:P131|wdt:P159) ?city. 
  filter not exists {?uni wdt:P576 ?date}. 
  ?city wdt:P17 wd:Q183. 
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en, de". } 
} 

## France

### Municipalities:

SELECT DISTINCT ?city ?cityLabel (MAX(?lat) AS ?latitude) (MAX(?long) AS ?longitude) (MAX(?pop) AS ?population) WHERE { 
  ?city (p:P31/ps:P31)/wdt:P279* wd:Q15284. 
  ?city wdt:P17 wd:Q142. 
  ?city wdt:P1082 ?pop. 
  filter(?pop >= 5000)
  ?city (p:P625/psv:P625) ?coordinates. 
  ?coordinates wikibase:geoLatitude ?lat .
  ?coordinates wikibase:geoLongitude ?long. 
  FILTER(?lat <=51.179 && ?lat >= 41.968 && ?long <= 8 && ?long >= -5.361) 
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en, fr"} 
} 
GROUP BY ?city ?cityLabel 
ORDER BY DESC(?population)

### Universities
SELECT DISTINCT ?city ?uni ?cityLabel ?uniLabel WHERE { 
  ?uni (p:P31/ps:P31)/wdt:P279* wd:Q3918. 
  ?uni (wdt:P131|wdt:P159) ?city. 
  filter not exists {?uni wdt:P576 ?date}.
  ?city wdt:P17 wd:Q142. 
  ?city (p:P625/psv:P625) ?coordinates. 
  ?coordinates wikibase:geoLatitude ?latitude. 
  ?coordinates wikibase:geoLongitude ?longitude. 
  FILTER(?latitude <=51.179 && ?latitude >= 41.968 && ?longitude <= 8 && ?longitude >= -5.361) 
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en, fr". } 
}


# Experiments

The resulting files are stored in the directories france and germany, having the names
**cities.json** and **universities_uncorrected.json**.
The handmade corrections are stored in the file **corrections.json**.
The **corrections.json** files are used to make the corrected file **universities.json**
in the following manner:

If [x,y,z] is in **corrections.json**, then the university Qx is located in Qy
in **universities_uncorrected.json**, but we replaced Qy by Qz in **universites.json**.

The experiments described in the paper can be reproduced by running **python3 main.py**.
Running main.py will:
- Generate summaries containing all relevant data for all municipalities.
- Compute the minimal step graph.
- Compute stats for the classification with normalized isolation, population and prominence values.
- Run the classification experiments described in the paper.
- Create the latex tabular containing the results.

The following external packages are needed:

- networkx (2.4)
- geopy (1.20.0)
- scikit-learn (0.21.3)
- pandas (0.25.2)
- numpy  (1.17.2)

The number in brackets are the versions that were used for the experiments.
We used Python version 3.7.4 for our experiments.
