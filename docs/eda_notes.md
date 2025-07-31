# EDA notes – water_co2.xlsx (apr_25 sheet)- example, will change

* **Nitrate / ammonium / phosphate**  
  *20 % blanks → set nullable in schema; treat NaNs separately in stats.*

* **Temp_wat outlier** at station J11 (31 °C @ 0 m) – double-check against CTD file.

* **Missing depth_m** rows: S14, P119. Could interpolate or drop before GAM.

(Keep adding bullet points as you spot issues.)
