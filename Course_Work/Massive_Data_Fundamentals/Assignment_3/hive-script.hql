DROP TABLE IF EXISTS rbigrams;

create external table rbigrams (
rbigram string,
year int,
occurrences int,
books int) 
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
LOCATION '/user/hadoop/r-bigrams/';

DROP TABLE IF EXISTS agg;

CREATE TABLE agg AS (
SELECT rbigram, 
SUM(occurrences) as occurrences,
SUM(books) AS books,
CAST(SUM(occurrences) AS FLOAT)/SUM(books) AS occurrences_per_book, 
MIN(year) AS first_appearance,
MAX(year) AS last_appearance,
COUNT(year) AS num_year
FROM rbigrams
GROUP BY rbigram);


INSERT OVERWRITE DIRECTORY 'hive_results' 
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ',' 
SELECT * 
FROM agg 
WHERE first_appearance = 1950 
AND last_appearance = 2009 
AND num_year = 60 
ORDER BY occurrences_per_book ASC 
LIMIT 50;
