using DataFrames
using CSV
using PyCall, JLD, PyCallJLD
using ArgParse


function main(args)

    # initialize the settings (the description is for the help screen)
    s = ArgParseSettings(description = "Example 1 for argparse.jl: minimal usage.")

    @add_arg_table s begin
        "--data"    
        #    arg_type = Int           # only Int arguments allowed
        #    nargs = '?'              # '?' means optional argument
        #    default = 0              # this is used when the option is not passed
        #    constant = 1             # this is used if --opt1 is paseed with no argument
            help = "Data file for train and test"
        
        "--output"
            help = "output model name"

        "choice"
            help = "Model choice"
    end

    parsed_args = parse_args(s) # the result is a Dict{String,Any}

    println(parsed_args["data"])
    println("Parsed args:")
    for (key,val) in parsed_args
        println("  $key  =>  $(repr(val))")
    end
end

main(ARGS)
