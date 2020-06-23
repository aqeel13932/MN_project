def Construct_Scenario(Scenario):
    Scenario = Scenario.split(',')
    result = ""
    for part in Scenario:
        morning_or_night=""
        if part[0]=='M':
            morning_or_night="1"
        else:
            morning_or_night="0"
        result+=morning_or_night*int(part[1:])
    return result
Scenarios={
    # Baseline (Random 4)
    0:("M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20"),
    #First morning longer 50%, Night smaller 50%
    1:("M30,N10,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20"),
    #First night longer 50%, morning smaller 50%
    2:("M10,N30,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20"),
    #Jetlag 2nd morning longer 50%
    3:("M20,N20,M30,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N10"),
    #Jet lag 2nd night longer 50%
    4:("M20,N20,M20,N30,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N10"),
    # Random 1
    5:("M10,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M10"),
    # Random 2
    6:("N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20"),
    # Random 3
    7:("N10,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N10"),
    #Jetlag 2nd morning longer 30%
    8:("M20,N20,M26,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N14"),
    #Jetlag 2nd night longer 30%
    9:("M20,N20,M20,N26,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N14"),
    #Jetlag 2nd morning longer 20%
    10:("M20,N20,M24,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N16"),
    #Jetlag 2nd morning longer 20%
    11:("M20,N20,M20,N24,M20,N20,M20,N20,M20,N20,M20,N20,M20,N20,M20,N16"),
    #All morning 50% more
    12:("M30,N20,M30,N20,M30,N20,M30,N20,M30,N20,M30,N20,M20"),
    #All nights 50% more
    13:("M20,N30,M20,N30,M20,N30,M20,N30,M20,N30,M20,N30,M20"),
    #All mornings 30% more
    14:("M26,N20,M26,N20,M26,N20,M26,N20,M26,N20,M26,N20,M26,N18"),
    #All nightss 30% more
    15:("M20,N26,M20,N26,M20,N26,M20,N26,M20,N26,M20,N26,M20,N24"),
    #All mornings 20% more
    16:("M24,N20,M24,N20,M24,N20,M24,N20,M24,N20,M24,N20,M24,N20,M12"),
    #All nights 20% more
    17:("M20,N24,M20,N24,M20,N24,M20,N24,M20,N24,M20,N24,M20,N24,M12"),
    # longer day (46 instead of 40)
    18:("M23,N23,M23,N23,M23,N23,M23,N23,M23,N23,M23,N23,M23,N21"),
    # shoerter day (17 instead of 40)
    19:("M17,N17,M17,N17,M17,N17,M17,N17,M17,N17,M17,N17,M17,N17,M17,N17,M17,N17,M14"),
    20:("M20,N20,M20,N20,M20,N20,M20,N20,N160"),
    21:("M20,N20,M20,N20,M20,N20,M20,N20,M160")
}
Scenarios_desc={
    0:"Baseline (Random 4)",
    1:"First morning longer 50%, Night smaller 50%",
    2:"First night longer 50%, morning smaller 50%",
    3:"Jetlag 2nd morning longer 50%",
    4:"Jet lag 2nd night longer 50%",
    5:"Random 1",
    6:"Random 2",
    7:"Random 3",
    8:"Jetlag 2nd morning longer 30%",
    9:"Jetlag 2nd night longer 30%",
    10:"Jetlag 2nd morning longer 20%",
    11:"Jetlag 2nd morning longer 20%",
    12:"All morning 50% more",
    13:"All nights 50% more",
    14:"All mornings 30% more",
    15:"All nights 30% more",
    16:"All mornings 20% more",
    17:"All nights 20% more",
    18:"longer day (46 instead of 40)",
    19:"shorter day (17 instead of 40)",
    20:"Last 4 days permenant night",
    21:"Last 4 days permenant mornings"
}
