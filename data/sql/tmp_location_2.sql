WITH LOCATION_DATA AS (
    SELECT *
    FROM `dataservices-asia-prod.location_db.vw_location_v2` 
    WHERE country_code = 'MY'
    AND location_data_source = 'MY_IPROPERTY'
    AND location_type IN ('COUNTRY', 'REGION', 'STATE', 'DIVISION', 'DISTRICT', 'CITY', 'POST_CODE', 'STREET_NAME', 'BUILDING_NAME')
), 

FLATTEN_LOC_DATA AS (
  SELECT 
  t.location_id, t.global_id, t.legacy_id, t.location_type, lower(t.location_name) location_name
  , CASE WHEN t.location_type="COUNTRY" THEN (SELECT value FROM UNNEST(t.alternate_name) WHERE language_code="en-GB" AND type="DISPLAY_NAME")
  ELSE (SELECT ANAME.value FROM UNNEST((SELECT ASLOC.alternate_name FROM UNNEST(associate_location) AS ASLOC WHERE ASLOC.location_type="COUNTRY")) AS ANAME WHERE ANAME.type="DISPLAY_NAME") END country_display_name
  , CASE WHEN t.location_type="REGION" THEN (SELECT value FROM UNNEST(t.alternate_name) WHERE language_code="en-GB" AND type="DISPLAY_NAME")
  ELSE (SELECT ANAME.value FROM UNNEST((SELECT ASLOC.alternate_name FROM UNNEST(associate_location) AS ASLOC WHERE ASLOC.location_type="REGION")) AS ANAME WHERE ANAME.type="DISPLAY_NAME") END region_display_name
  , CASE WHEN t.location_type="STATE" THEN (SELECT value FROM UNNEST(t.alternate_name) WHERE language_code="en-GB" AND type="DISPLAY_NAME")
  ELSE (SELECT ANAME.value FROM UNNEST((SELECT ASLOC.alternate_name FROM UNNEST(associate_location) AS ASLOC WHERE ASLOC.location_type="STATE")) AS ANAME WHERE ANAME.type="DISPLAY_NAME") END state_display_name
  , CASE WHEN t.location_type="CITY" THEN (SELECT value FROM UNNEST(t.alternate_name) WHERE language_code="en-GB" AND type="DISPLAY_NAME")
  ELSE (SELECT ANAME.value FROM UNNEST((SELECT ASLOC.alternate_name FROM UNNEST(associate_location) AS ASLOC WHERE ASLOC.location_type="CITY")) AS ANAME WHERE ANAME.type="DISPLAY_NAME") END city_display_name
  , CASE WHEN t.location_type="POST_CODE" THEN (SELECT value FROM UNNEST(t.alternate_name) WHERE language_code="en-GB" AND type="DISPLAY_NAME")
  ELSE (SELECT ANAME.value FROM UNNEST((SELECT ASLOC.alternate_name FROM UNNEST(associate_location) AS ASLOC WHERE ASLOC.location_type="POST_CODE")) AS ANAME WHERE ANAME.type="DISPLAY_NAME") END postcode_display_name
  , CASE WHEN t.location_type="DIVISION" THEN (SELECT value FROM UNNEST(t.alternate_name) WHERE language_code="en-GB" AND type="DISPLAY_NAME")
  ELSE (SELECT ANAME.value FROM UNNEST((SELECT ASLOC.alternate_name FROM UNNEST(associate_location) AS ASLOC WHERE ASLOC.location_type="DISTRICT")) AS ANAME WHERE ANAME.type="DISPLAY_NAME") END district_display_name
  , CASE WHEN t.location_type="DIVISION" THEN (SELECT value FROM UNNEST(t.alternate_name) WHERE language_code="en-GB" AND type="DISPLAY_NAME")
  ELSE (SELECT ANAME.value FROM UNNEST((SELECT ASLOC.alternate_name FROM UNNEST(associate_location) AS ASLOC WHERE ASLOC.location_type="DIVISION")) AS ANAME WHERE ANAME.type="DISPLAY_NAME") END div_display_name
  , CASE WHEN t.location_type="STREET_NAME" THEN (SELECT value FROM UNNEST(t.alternate_name) WHERE language_code="en-GB" AND type="DISPLAY_NAME")
  ELSE (SELECT ANAME.value FROM UNNEST((SELECT ASLOC.alternate_name FROM UNNEST(associate_location) AS ASLOC WHERE ASLOC.location_type="STREET_NAME")) AS ANAME WHERE ANAME.type="DISPLAY_NAME") END str_display_name
  , CASE WHEN t.location_type="BUILDING_NAME" THEN (SELECT value FROM UNNEST(t.alternate_name) WHERE language_code="en-GB" AND type="DISPLAY_NAME")
  ELSE (SELECT ANAME.value FROM UNNEST((SELECT ASLOC.alternate_name FROM UNNEST(associate_location) AS ASLOC WHERE ASLOC.location_type="BUILDING_NAME")) AS ANAME WHERE ANAME.type="DISPLAY_NAME") END bld_display_name
  FROM LOCATION_DATA t
),

SEARCH_ANALYTICS AS (
  SELECT clientId
  , DATETIME(TIMESTAMP_SECONDS(visitStartTime), "Asia/Kuala_Lumpur") visitMYDateTime
  , visitId
  , visitNumber
  , hitNumber
  , filter_propertyType
  , filter_listingType
  , filter_tenures
  , filter_furnishing
  , filter_market
  , filter_location
  , filter_location1
  , filter_location2
  , filter_location3
  , GeneralKeyword
  , PropertyKeyword
  , DeveloperKeyword
  , DevelopmentKeyword
  , PlaceKeyword
  , CASE
      WHEN GeneralKeyword IS NOT NULL THEN REPLACE(REPLACE(LOWER(GeneralKeyword), '  ', ' '), '+', ' ')
      WHEN PropertyKeyword IS NOT NULL THEN REPLACE(LOWER(PropertyKeyword), '+', ' ')
      WHEN DevelopmentKeyword IS NOT NULL THEN REPLACE(LOWER(DevelopmentKeyword), '+', ' ')
      WHEN DeveloperKeyword IS NOT NULL THEN REPLACE(LOWER(DeveloperKeyword), '+', ' ')
      WHEN filter_location3 IS NOT NULL AND filter_location3 NOT IN ('All','') THEN REPLACE(REPLACE(filter_location3, '-', ' '), '+', ' ')
      WHEN filter_location2 IS NOT NULL AND filter_location2 NOT IN ('All','') THEN REPLACE(REPLACE(filter_location2, '-', ' '), '+', ' ')
      WHEN filter_location1 IS NOT NULL AND filter_location1 NOT IN ('All','') THEN REPLACE(REPLACE(filter_location1, '-', ' '), '+', ' ')
    END Keyword
  FROM `data-services-asia-staging.raw_ga.search_web` 
  WHERE Date >= "2020-11-04"
  AND Market="MY"
  AND (
    # filter_location1 = Region
    # filter_location2 = City
    # filter_location3 = Area
    NOT (filter_location1="All" AND filter_location2="All" AND filter_location3="All")
    OR NOT (NULLIF(TRIM(PropertyKeyword), "") IS NULL AND NULLIF(TRIM(DeveloperKeyword), "") IS NULL AND NULLIF(TRIM(DevelopmentKeyword), "") IS NULL)
    OR NOT NULLIF(TRIM(PlaceKeyword), "") IS NULL
  )
),

STD_LOCATION AS (
  SELECT s.*
  , COALESCE(lev3.level1, lev2.level1, lev1.level1) level_1
  , COALESCE(lev3.level2, lev2.level2) level_2
  , COALESCE(lev3.level3) level_3
  FROM SEARCH_ANALYTICS s
  LEFT JOIN `regional-reporting.Malaysia_Production.AutoCompleteForSearchMapping` lev1
    ON s.Keyword = lev1.level1 AND lev1.FirstOrder = 1
  LEFT JOIN `regional-reporting.Malaysia_Production.AutoCompleteForSearchMapping` lev2
    ON s.Keyword = lev2.level2 AND lev2.FirstOrder = 2
  LEFT JOIN
    `regional-reporting.Malaysia_Production.AutoCompleteForSearchMapping` lev3
  ON s.Keyword = lev3.level3 AND lev3.FirstOrder = 3 AND (filter_location3 != '' OR GeneralKeyword != '' OR PropertyKeyword != '')
),

USABLE_LOC AS (
  SELECT * EXCEPT(Usable)
  FROM (
    SELECT
      *,
      CASE
        WHEN REPLACE(filter_location1,'-',' ') = level_1 THEN TRUE
        WHEN REPLACE(filter_location2,'-',' ') = level_2 THEN
          CASE
            WHEN REPLACE(filter_location1,'-',' ') = level_1 THEN TRUE
            WHEN filter_location1 IS NULL OR filter_location1 = '' THEN TRUE
            ELSE FALSE
          END
        WHEN REPLACE(filter_location3,'-',' ') = level_3 THEN
          CASE
            WHEN REPLACE(filter_location1,'-',' ') = level_1 THEN TRUE
            WHEN filter_location1 IS NULL OR filter_location1 = '' THEN TRUE
            ELSE FALSE
          END
        WHEN PropertyKeyword IS NOT NULL AND LOWER(PropertyKeyword) = level_3 THEN TRUE
        WHEN DevelopmentKeyword IS NOT NULL AND LOWER(DevelopmentKeyword) = level_3 THEN TRUE
        WHEN DeveloperKeyword IS NOT NULL THEN TRUE
        WHEN GeneralKeyword IS NOT NULL THEN TRUE
        ELSE FALSE
      END Usable
    FROM STD_LOCATION
  ) WHERE Usable IS TRUE
), 


-- filter_location1 = Region
-- filter_location2 = City
-- filter_location3 = Area

SEARCH_COMPILED AS (
  SELECT 
    stage.* EXCEPT(level_1, level_2, level_3),
    COALESCE(special_1.level1, special_2.level1, stage.level_1) level_1,
    COALESCE(special_2.level2, stage.level_2) level_2,
    stage.level_3
  FROM USABLE_LOC stage
  LEFT JOIN
    `regional-reporting.Malaysia_Production.AutoCompleteForSearchMapping` special_2
  ON REPLACE(filter_location3,'-',' ') = special_2.level2 AND special_2.FirstOrder = 2 AND stage.level_1 IS NULL AND stage.level_2 IS NULL AND stage.level_3 IS NULL AND filter_location3 IS NOT NULL
  LEFT JOIN
    `regional-reporting.Malaysia_Production.AutoCompleteForSearchMapping` special_1
  ON REPLACE(filter_location1,'-',' ') = special_1.level1 AND special_1.FirstOrder = 1 AND stage.level_1 IS NULL AND stage.level_2 IS NULL AND stage.level_3 IS NULL AND filter_location1 IS NOT NULL AND filter_location3 IS NULL
)

-- 494760
-- SELECT clientId, visitId, visitNumber, hitNumber, COUNT(1)
-- FROM (
  SELECT DISTINCT A.*
  , COALESCE(B.bld_display_name, C.bld_display_name, D.bld_display_name) map_location_3
  , COALESCE(B.city_display_name, C.city_display_name, D.city_display_name) map_location_2
  , COALESCE(B.state_display_name, C.state_display_name, D.state_display_name) map_location_1
  , COALESCE(B.global_id, C.global_id, D.global_id) global_id
  , COALESCE(B.legacy_id, C.legacy_id, D.legacy_id) legacy_id
  , COALESCE(B.location_type, C.location_type, D.location_type) location_type
  , COALESCE(B.location_name, C.location_name, D.location_name) display_name
  FROM SEARCH_COMPILED A
  LEFT JOIN FLATTEN_LOC_DATA B ON (A.level_3 = LOWER(B.bld_display_name)) AND (A.level_2 = LOWER(B.city_display_name)) AND (A.level_1 = LOWER(B.state_display_name))
  LEFT JOIN FLATTEN_LOC_DATA C ON (A.level_2 = LOWER(C.city_display_name)) AND (A.level_1 = LOWER(C.state_display_name))
  LEFT JOIN FLATTEN_LOC_DATA D ON (A.level_1 = LOWER(D.state_display_name))
  WHERE (COALESCE(
          IF(NULLIF(TRIM(level_3), "") IS NULL, NULL, "BUILDING_NAME"),
          IF(NULLIF(TRIM(level_2), "") IS NULL, NULL, "CITY"),
          IF(NULLIF(TRIM(level_1), "") IS NULL, NULL, "STATE")
          ) = COALESCE(B.location_type, C.location_type, D.location_type))
  AND (IF(filter_location1!="All", lower(filter_location1), lower(A.level_1)) = lower(A.level_1))
  AND (IF(filter_location2!="All", lower(filter_location2), lower(A.level_2)) = lower(A.level_2))
  AND (IF(filter_location3!="All", lower(filter_location3), lower(A.level_3)) = lower(A.level_3))
--   AND clientId="102637396.1591089552" AND visitId=1604501916 AND visitNumber=37 AND hitNumber=7
-- ) GROUP BY 1,2,3,4 ORDER BY 5 DESC