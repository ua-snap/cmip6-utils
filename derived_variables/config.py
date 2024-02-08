# min / max possible ranges
expected_value_ranges = {
    # this value was determined in exploration
    #  max temps for panarctic domain were ~45 C, min temps around -60 C
    #  within day differences of 100 C are extremely unlikely
    #  (units for data are kelvin but )
    "dtr": {"minimum": 0, "maximum": 100}
}
