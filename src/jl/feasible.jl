function sample_feasible_u(G, T, u0, U0, L0, Ton, Toff)
    u = zeros(Int, G, T)
    for g in 1:G
        t = 1
        # Initial block
        if u0[g] == 1 && U0[g] > 0
            for tt in t:min(t+U0[g]-1, T)
                u[g, tt] = 1
            end
            t += U0[g]
        elseif u0[g] == 0 && L0[g] > 0
            for tt in t:min(t+L0[g]-1, T)
                u[g, tt] = 0
            end
            t += L0[g]
        end
        # Fill the rest
        while t <= T
            state = rand([0, 1])
            if state == 1
                duration = rand(Ton[g]:max(Ton[g], T-t+1))
                for tt in t:min(t+duration-1, T)
                    u[g, tt] = 1
                end
                t += duration
            else
                duration = rand(Toff[g]:max(Toff[g], T-t+1))
                for tt in t:min(t+duration-1, T)
                    u[g, tt] = 0
                end
                t += duration
            end
        end
    end
    return u
end

G = 3
T = 24
u0 = [0, 0, 0]  # Initial states for each unit
U0 = [0, 0, 0]  # Minimum up times for each
L0 = [0, 0, 0]  # Minimum down times for each
Ton = [4, 3, 1]  # Minimum up times for each unit
Toff = [4, 3, 0]  # Minimum down times for each unit
u = sample_feasible_u(G, T, u0, U0, L0, Ton, Toff)
println(u)
