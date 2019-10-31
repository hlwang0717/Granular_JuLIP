using JuLIP, PyPlot, PyCall

function myplot(at, ttl = nothing)
    x, y, _ = xyz(at)
    r = at.M
    N = length(at.Z)
    color = Array{String}(N)
    for i = 1:N
        z = at.Z[i];
        if z == 1 
            color[i] = "blue";
        else
            color[i] = "red";
        end
    #    PyPlot.plot(mod(x[i],1), mod(y[i],1), color, markersize=20) #set to be mod(x, Lx), mod(y, Ly)
    end
    if ttl != nothing
        title(ttl)
    end 
    
    cell_x = [0.0, 1.0, 1.0, 0.0, 0.0]
    cell_y = [0.0, 0.0, 1.0, 1.0, 0.0]
    
    shape = [circle(x0=mod(x[i],1)-r[i], y0=mod(y[i],1)-r[i], 
                    x1=mod(x[i],1)+r[i], y1=mod(y[i],1)+r[i];
                    opacity = 0.8, fillcolor=color[i], line_color="transparent")
             for i in 1:N]
    
    PlotlyJS.plot(cell_x, cell_y, Layout(;height=600,width=600,showlegend=false,shapes=shape))
    
end 

