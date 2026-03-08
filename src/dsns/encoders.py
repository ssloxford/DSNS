# TODO

from bitstring import BitStream

from dsns.helpers import SatID
from dsns.presets import MultiConstellation

def source_route_encode(path: list[SatID], mobility: MultiConstellation, planes: int=72, sats_per_plane:int=22) -> BitStream:
    intra_plane_bit = "uint:1=0"
    intra_plane_bwd = "uint:1=0"
    intra_plane_fwd = "uint:1=1"
    inter_plane_bit = "uint:1=1"
    plane = "uint7:="
    plane_index = "uint5:="

    header = BitStream()
    intra_hops = False
    intra_hops_count = 0
    for i in range(len(path) - 1):
        sat_id_1, sat_id_2 = path[i], path[i + 1]
        sat_1 = mobility.satellites.by_id(sat_id_1)
        sat_2 = mobility.satellites.by_id(sat_id_2)

        if sat_1 is None or sat_2 is None:
            raise Exception("Invalid satellites in path")

        _, plane_1, ind_1 = sat_1.name.split("_")
        _, plane_2, ind_2 = sat_2.name.split("_")

        if plane_1 == plane_2:
            if not intra_hops:
                header.append(intra_plane_bit)
                intra_hops = True
                if (int(ind_1) + 1) % sats_per_plane == int(ind_2):
                    header.append(intra_plane_fwd)
                else:
                    header.append(intra_plane_bwd)
            intra_hops_count += 1
        else:
            if intra_hops:
                intra_hops = False
                header.append(plane_index+str(intra_hops_count))
                intra_hops_count = 0
            header.append(inter_plane_bit)
            header.append(plane+plane_2)
            header.append(plane_index+ind_2)
    if intra_hops:
        header.append(plane_index + str(intra_hops_count))

    return header

def source_route_read_next_segment(header: BitStream, curr_sat_id: SatID, mobility: MultiConstellation, planes: int=72, sats_per_plane: int=22) -> SatID:
    start_pos = header.pos
    is_interplane = header.read("bool")

    if is_interplane:
        plane = int(header.read("uint:7"))
        index = int(header.read("uint:5"))
        sat_name = f"starlink_{plane}_{index}"
        curr_sat = mobility.satellites.by_name(sat_name)

        if curr_sat is None:
            raise Exception("Invalid path encoding supplied")

        return curr_sat.sat_id
    else:
        curr_sat = mobility.satellites.by_id(curr_sat_id)

        if curr_sat is None:
            raise Exception("Invalid path encoding supplied")

        _, plane, ind = curr_sat.name.split("_")
        is_forward = header.read("bool")
        hops = int(header.read("uint:5"))
        inc = 1 if is_forward else - 1

        sat_name = f"starlink_{plane}_{(int(ind) + inc) % sats_per_plane}"

        curr_sat = mobility.satellites.by_name(sat_name)

        if curr_sat is None:
            raise Exception("Invalid path encoding supplied")

        if hops - 1 != 0:
            header.overwrite(f"uint:5={hops - 1}", pos=header.pos-5)
            header.pos = start_pos

        return curr_sat.sat_id

def source_route_decode(header: BitStream, source: SatID, mobility: MultiConstellation, planes: int=72, sats_per_plane: int=22) -> list[SatID]:
    path = [source]
    curr_sat_id = source
    
    while header.pos < header.len:
        is_interplane = header.read("bool")

        if is_interplane:
            plane = int(header.read("uint:7"))
            index = int(header.read("uint:5"))
            sat_name = f"starlink_{plane}_{index}"
            curr_sat = mobility.satellites.by_name(sat_name)

            if curr_sat is None:
                raise Exception("Invalid path encoding supplied")

            path.append(curr_sat.sat_id)
        else:
            curr_sat = mobility.satellites.by_id(curr_sat_id)

            if curr_sat is None:
                raise Exception("Invalid path encoding supplied")

            _, plane, ind = curr_sat.name.split("_")
            is_forward = header.read("bool")
            hops = int(header.read("uint:5"))
            inc = 1 if is_forward else - 1

            for i in range(1, hops + 1):
                sat_name = f"starlink_{plane}_{(int(ind) + i * inc) % sats_per_plane}"

                curr_sat = mobility.satellites.by_name(sat_name)

                if curr_sat is None:
                    raise Exception("Invalid path encoding supplied")

                curr_sat_id = curr_sat.sat_id
                path.append(curr_sat.sat_id)

    return path

