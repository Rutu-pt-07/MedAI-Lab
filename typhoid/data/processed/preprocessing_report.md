# Preprocessing Report

## Row Counts
| Stage | Count |
|-------|-------|
| Raw dataset | 1106 |
| Corrupt rows removed | 6 |
| Final cleaned dataset | 1100 |

## Column Data Types
| Column | Dtype |
|--------|-------|
| Gender | int64 |
| TO | float64 |
| TH | float64 |
| AH | float64 |
| BH | float64 |
| OX2 | float64 |
| OXK | float64 |
| OX9 | float64 |
| A | float64 |
| M | float64 |
| Rickettsia_Suspect | float64 |
| Acute_typhoid | int64 |
| Paratyphoid_A | int64 |
| Paratyphoid_B | int64 |
| Typhoid | int64 |
| Age_years | float64 |
| rickettsia_panel_conducted | int64 |

## NaN Counts (Post-Cleaning)
| Column | NaN Count | % Missing |
|--------|-----------|-----------|
| Gender | 0 | 0.0% |
| TO | 0 | 0.0% |
| TH | 0 | 0.0% |
| AH | 0 | 0.0% |
| BH | 0 | 0.0% |
| OX2 | 0 | 0.0% |
| OXK | 0 | 0.0% |
| OX9 | 0 | 0.0% |
| A | 48 | 4.4% |
| M | 48 | 4.4% |
| Rickettsia_Suspect | 397 | 36.1% |
| Acute_typhoid | 0 | 0.0% |
| Paratyphoid_A | 0 | 0.0% |
| Paratyphoid_B | 0 | 0.0% |
| Typhoid | 0 | 0.0% |
| Age_years | 0 | 0.0% |
| rickettsia_panel_conducted | 0 | 0.0% |

## Rickettsia Panel Breakdown
| Status | Count |
|--------|-------|
| Panel conducted | 709 |
| Panel NOT conducted | 397 |

## Age Statistics
| Stat | Value |
|------|-------|
| Min | 0.001 years |
| Max | 90.0 years |
| Mean | 29.7 years |
| Median | 25.0 years |

## Titer Value Distributions

### TO
| Value | Count |
|-------|-------|
| 1:80 | 784 |
| 1:160 | 312 |
| 1:320 | 4 |

### TH
| Value | Count |
|-------|-------|
| 1:80 | 737 |
| 1:160 | 328 |
| 1:320 | 35 |

### AH
| Value | Count |
|-------|-------|
| 1:80 | 1019 |
| 1:160 | 41 |
| 1:320 | 40 |

### BH
| Value | Count |
|-------|-------|
| 1:80 | 1075 |
| 1:160 | 24 |
| 1:320 | 1 |

### OX2
| Value | Count |
|-------|-------|
| 0 (not tested) | 397 |
| 1:80 | 512 |
| 1:160 | 131 |
| 1:320 | 60 |

### OXK
| Value | Count |
|-------|-------|
| 0 (not tested) | 397 |
| 1:80 | 489 |
| 1:160 | 160 |
| 1:320 | 54 |

### OX9
| Value | Count |
|-------|-------|
| 0 (not tested) | 397 |
| 1:80 | 631 |
| 1:160 | 57 |
| 1:320 | 15 |

### A
| Value | Count |
|-------|-------|
| 0 (not tested) | 397 |
| 1:80 | 655 |

### M
| Value | Count |
|-------|-------|
| 0 (not tested) | 397 |
| 1:80 | 629 |
| 1:160 | 26 |

## Target Variable Distributions

### Typhoid (Primary Target)
| Class | Count | % |
|-------|-------|---|
| Negative (0) | 462 | 42.0% |
| Minimal (1) | 560 | 50.9% |
| Positive (2) | 78 | 7.1% |

### Acute_typhoid
| Class | Count |
|-------|-------|
| 0 | 784 |
| 1 | 316 |

### Paratyphoid_A
| Class | Count |
|-------|-------|
| 0 | 1019 |
| 1 | 81 |

### Paratyphoid_B
| Class | Count |
|-------|-------|
| 0 | 1075 |
| 1 | 25 |

### Rickettsia_Suspect
| Class | Count |
|-------|-------|
| 0 | 424 |
| 1 | 279 |
| NaN | 397 |
