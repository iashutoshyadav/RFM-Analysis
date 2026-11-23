-- RFM v2 (PostgreSQL) - matches the Python pipeline in rfm/.
-- Improvements over processing_query.sql:
--   * excludes cancelled invoices ('C' prefix) and exact duplicate rows
--   * snapshot date = day after last transaction (most recent buyer -> recency 1)
--   * R x FM grid segmentation: every score combination gets a named segment,
--     so nobody falls into "Other"
--   * outputs average order value and a recommended action per customer

WITH clean AS (
    SELECT DISTINCT *
    FROM online_retail
    WHERE customer_id IS NOT NULL
      AND invoice_no NOT LIKE 'C%'
      AND quantity > 0
      AND unit_price > 0
),
snapshot AS (
    SELECT MAX(invoice_date)::date + 1 AS snap FROM clean
),
rfm_raw AS (
    SELECT
        c.customer_id,
        s.snap - MAX(c.invoice_date)::date          AS recency,
        COUNT(DISTINCT c.invoice_no)                AS frequency,
        ROUND(SUM(c.quantity * c.unit_price), 2)    AS monetary
    FROM clean c CROSS JOIN snapshot s
    GROUP BY c.customer_id, s.snap
),
scored AS (
    SELECT *,
        NTILE(5) OVER (ORDER BY recency DESC, customer_id)  AS r_score,
        NTILE(5) OVER (ORDER BY frequency, customer_id)     AS f_score,
        NTILE(5) OVER (ORDER BY monetary, customer_id)      AS m_score
    FROM rfm_raw
),
graded AS (
    SELECT *,
        (f_score + m_score + 1) / 2 AS fm_score   -- mean of F and M, rounded half up
    FROM scored
)
SELECT
    customer_id, recency, frequency, monetary,
    ROUND(monetary / frequency, 2) AS avg_order_value,
    r_score, f_score, m_score, fm_score,
    CONCAT(r_score, f_score, m_score) AS rfm_code,
    CASE
        WHEN r_score = 5 THEN (ARRAY['New Customers','Promising','Potential Loyalists','Loyal Customers','Champions'])[fm_score]
        WHEN r_score = 4 THEN (ARRAY['Promising','Potential Loyalists','Potential Loyalists','Loyal Customers','Champions'])[fm_score]
        WHEN r_score = 3 THEN (ARRAY['About to Sleep','Need Attention','Need Attention','Loyal Customers','Loyal Customers'])[fm_score]
        WHEN r_score = 2 THEN (ARRAY['Hibernating','About to Sleep','At Risk','At Risk','Cannot Lose Them'])[fm_score]
        ELSE                  (ARRAY['Lost','Hibernating','At Risk','At Risk','Cannot Lose Them'])[fm_score]
    END AS segment
FROM graded
ORDER BY customer_id;
