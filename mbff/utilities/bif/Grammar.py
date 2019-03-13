bif_grammar = """
%import common.ESCAPED_STRING
%import common.LETTER
%import common.DIGIT
%import common.DECIMAL
%import common.INT
%import common.WS
%ignore WS

?start : network_definition

network_definition : block*
?block: network_block
    | variable_block
    | probability_block

network_block : "network" network_name "{" property_list "}"
variable_block : "variable" variable_identifier "{" variable_type_def  property_list "}"
probability_block : "probability" "(" variable_identifier ("|" conditioning_set)? ")" "{" variable_value_probabilities property_list "}"

identifier : /[a-zA-Z0-9-_]+/
string_value : /[a-zA-Z0-9-_.]+/
property_value : (ESCAPED_STRING | string_value)
property : "property" identifier property_value ";"
property_list : property*
network_name : identifier

variable_identifier : identifier
?variable_value : identifier

variable_list : variable_identifier ("," variable_identifier)*
variable_value_list : variable_value ("," variable_value)*
probability_value : DECIMAL
probabilities_list : probability_value ("," probability_value)*


variable_value_count : INT
variable_type_def : "type" "discrete" "[" variable_value_count "]" "{" variable_value_list "}" ";"

conditioning_set : variable_list
variable_value_tuple : "(" variable_value_list ")"
variable_value_probabilities : variable_value_probabilities__simple | variable_value_probabilities__cond
variable_value_probabilities__simple : "table" probabilities_list ";"
variable_value_probabilities__cond : variable_value_probabilities__cond_line+
variable_value_probabilities__cond_line : variable_value_tuple probabilities_list ";"
"""
