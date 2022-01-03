# Multi Model fusion test:

## Environments

It loads CSV files of the LVMS, transfer it from latitute longtitude into ENU (U is 0), and then calculate the distance from the start line for each point, then append the distance to the [:,2] column.

It then generates a point moving along the path, and calculate the X, Y coordinates of the point in ENU from a function which loop through the list of the look up table, and use _map() helper function to get a weighted average of the X, Y coordinates in the list, to get the X, Y coordinates of the point.

The look up table is like this:
~~~
    enu_mat_in = :
        [[x,y,distSF_in]
        [x,y,distSF_in]
        [x,y,distSF_in]
        ...
        [x,y,distSF_in]]
~~~

## TODO : fit polynomial functions to replace the look up table

the input of the function should be the distance from the start line, and the output should be the X, Y coordinates of the point in ENU, for inner and outer ring of the track repectively. Thus, there are 4 curves to fit.

If you can, can you fit a periodic function to the look up table? So we can get the correct X, Y coordinates even if the distance is out of the range.