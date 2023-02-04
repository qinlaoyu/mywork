WITH relevant_total_counts AS (
  SELECT TO_TIMESTAMP(zone_status.tstamp) AS zone_status_tstamp,
         total_count.count_enter AS entered FROM zone_status
  LEFT JOIN total_count ON total_count.zone_status_id=zone_status.id
  LEFT JOIN zone ON zone_status.zone_id=zone.id
  LEFT JOIN stream_configuration ON stream_configuration.id=zone.stream_id
  WHERE
    total_count.class_name=$class_name AND
    zone.name=SPLIT_PART('$zone_name', ': ', '2') AND
    stream_configuration.name=SPLIT_PART('$zone_name', ': ', '1') AND
    zone_status.id >= (SELECT id FROM zone_status WHERE tstamp >= $__unixEpochFrom() ORDER BY tstamp ASC LIMIT 1) AND
    zone_status.id <= (SELECT id FROM zone_status WHERE tstamp < $__unixEpochTo() ORDER BY tstamp DESC LIMIT 1) 
)

SELECT DATE_TRUNC('$time_bin', zone_status_tstamp) AS time,
  MAX(entered) AS total,
  MAX(entered) - MIN(entered) AS entered FROM relevant_total_counts
GROUP BY 
  time
ORDER BY time
