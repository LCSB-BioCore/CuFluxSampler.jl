
using Downloads
using SHA

function check_data_file_hash(path, expected_checksum)
    actual_checksum = bytes2hex(sha256(open(path)))
    if actual_checksum != expected_checksum
        @error "The downloaded data file `$path' seems to be different from the expected one. Tests will likely fail." actual_checksum expected_checksum
    end
end

function download_data_file(url, path, hash)
    if isfile(path)
        check_data_file_hash(path, hash)
        @info "using cached `$path'"
        return path
    end

    Downloads.download(url, path)
    check_data_file_hash(path, hash)
    return path
end


isdir("downloaded") || mkdir("downloaded")
df(s) = joinpath("downloaded", s)

model_paths = Dict{String,String}(
    "e_coli_core.xml" => download_data_file(
        "http://bigg.ucsd.edu/static/models/e_coli_core.xml",
        df("e_coli_core.xml"),
        "b4db506aeed0e434c1f5f1fdd35feda0dfe5d82badcfda0e9d1342335ab31116",
    ),
)
