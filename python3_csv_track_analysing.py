"""
A simple example of an animated plot... In 3D!
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

#for coordinate conversion to utm
import utm

#for operations
import os
#import math

#for kernal creation
#import geopandas
#import geoplot

class PARTTracks(object):

    def __init__(self, folder=None):
      
        self.folder = folder
        self.tracks = None
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.zmin = None
        self.zmax = None

        self.columnnames = ["xa", "ya", "zdepth", "sdepth","zlevel",\
                            "stemp","dtemp","duration","vzact","wsettl","istage"]

        if(folder != None):
            self.tracks = self.read_tracks_from_csv()
            self.xmin = np.nanmin(self.tracks.xa.astype('float64')) ; self.xmax = np.nanmax(self.tracks.xa.astype('float64'))
            self.ymin = np.nanmin(self.tracks.ya.astype('float64')) ; self.ymax = np.nanmax(self.tracks.ya.astype('float64'))
            self.zmin = np.nanmax(self.tracks.zdepth.astype('float64'))*-1 ; self.zmax = np.nanmin(self.tracks.zdepth.astype('float64'))*-1
            self.timemin = np.nanmin(self.tracks.Timestep) - np.nanmin(self.tracks.Timestep) 
            self.timemax = np.nanmax(self.tracks.Timestep) - np.nanmin(self.tracks.Timestep)


            self.individuals    = self.tracks.Individual.unique()
            self.timesteps      = self.tracks.Timestep.unique() - self.tracks.Timestep.min()
            


    def Gen_TrackLine(self, data, timesteps, individual_nr) :
        """
        Create a track line using the start and end coordinates

        """
        dims = 6
        lineData = np.empty((dims, len(timesteps)))
        lineData[:, 0] = (float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'))
        for index, timestep in enumerate(timesteps) :
            #Extract the xy positions per timestep
            if(len(data.loc[(data['Individual'] == individual_nr) & (data['Timestep'] == timestep)]) != 0):
                x = data.loc[(data['Individual'] == individual_nr) & (data['Timestep'] == timestep)]['xa'].astype('float64').values.tolist()[0]
                y = data.loc[(data['Individual'] == individual_nr) & (data['Timestep'] == timestep)]['ya'].astype('float64').values.tolist()[0]
                z = -1 * data.loc[(data['Individual'] == individual_nr) & (data['Timestep'] == timestep)]['zdepth'].astype('float64').values.tolist()[0]
                tempavr     = data.loc[(data['Individual'] == individual_nr) & (data['Timestep'] == timestep)]['dtemp'].astype('int64').values.tolist()[0]
                bottomdepth = z + -1 * data.loc[(data['Individual'] == individual_nr) & (data['Timestep'] == timestep)]['zlevel'].astype('float64').values.tolist()[0] 
                stage       = data.loc[(data['Individual'] == individual_nr) & (data['Timestep'] == timestep)]['istage'].astype('int64').values.tolist()[0]
            else:
                x = float('nan'); y = float('nan'); z = float('nan')
                tempavr = float('nan'); bottomdepth = float('nan'); stage = float('nan')

            
            step = [x,y,z, tempavr, bottomdepth, stage]
            
            #Set end of model to nan, else it is somewhere far away
            if([step[0],step[1]] == [-1000,-1000]):
                x = float('nan'); y = float('nan'); z = float('nan')
                tempavr = float('nan'); bottomdepth = float('nan'); stage = float('nan')
                step = [x,y,z, tempavr, bottomdepth, stage]

            lineData[:, index] = step

        return(lineData)

    def update_lines(self, num, dataLines, lines) :
        for line, data in zip(lines, dataLines) :
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(data[0:2, :num])
            line.set_3d_properties(data[2,:num])
        return(lines)

    def read_tracks_from_csv(self):
        appended_tracks = pd.DataFrame([])
        track_files = self.find_csv_filenames(self.folder, suffix = ".csv")
        track_files.sort()
        
        for nr, track_file in enumerate(track_files):
            try:
                track_data = pd.read_csv(self.folder + track_file, header=None, names = self.columnnames)
            except:
                print("Warning - could not read track file : " + track_file)
                continue

            #Store individual number (rownr) and timestep (csvnr)
            track_data['Individual'] = track_data.reset_index().index + 1
            track_data['Timestep'] = nr

            #make correct data format
            to_int = ['Individual','Timestep'] + [x for x in self.columnnames if x == "istage"]
            to_float = [x for x in self.columnnames if x != "istage"]

            #add nans where applicable
            track_data[to_float] = track_data[to_float].replace(-1000.0,np.NaN)
            track_data[to_float] = track_data[to_float].replace(-999.00,np.NaN)
            track_data[["xa","ya"]] = track_data[["xa","ya"]].replace(0.0,np.NaN)
            track_data[["xa","ya","duration"]] = track_data[["xa","ya","duration"]].replace("**********",np.NaN)

            #make appropriate data type
            try:
                track_data[to_int] = track_data[to_int].astype('int64')
            except:
                raise ValueError("Could not convert columns to integers : " + self.folder + track_file)
            
            try:
                track_data[to_float] = track_data[to_float].astype('float64')
            except:
                raise ValueError("Could not convert columns to floats : " + self.folder + track_file)
            
            #Add together
            appended_tracks = appended_tracks.append(track_data)

        return(appended_tracks)


    def find_csv_filenames(self, path_to_dir, suffix=".csv" ):
        filenames = os.listdir(path_to_dir)
        return [ filename for filename in filenames if filename.endswith( suffix ) ]

    def Setup_plot_boundaries(self, tracks, timesteps, individuals, xmin, xmax, ymin,\
                                 ymax, zmin, zmax):
        if(tracks == None):
            tracks = self.tracks
        if(timesteps == None):
            timesteps = self.timesteps
        if(individuals == None):
            individuals = self.individuals
        if(xmin == None):
            xmin = self.xmin
        if(xmax == None):
            xmax = self.xmax
        if(ymin == None):
            ymin = self.ymin
        if(ymax == None):
            ymax = self.ymax
        if(zmin == None):
            zmin = self.zmin
        if(zmax == None):
            zmax = self.zmax

        return((tracks, timesteps, individuals, xmin, xmax, ymin, ymax, zmin, zmax))

    def Setup_visualize_TrackLine(self, fig, ax, tracks, timesteps,individuals, xmin,\
                                     xmax, ymin, ymax, zmin, zmax) :

        data = [self.Gen_TrackLine(tracks,timesteps = timesteps, individual_nr = individual) for individual in individuals]

        # Creating fifty line objects.
        # NOTE: Can't pass empty arrays into 3d version of plot()
        lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

        return((fig, ax, data, lines))


    def Static_particle_characteristics_vs_time_verticle(self, tracks = None, timesteps = None, individual = None, timemin = None, \
        timemax = None, zmin = None, zmax = None, stagemin = None, stagemax = None, timestep = 1):

        if(not(isinstance(individual,int))):
            raise ValueError("Individual should be one individual nr")

        if(individual == None):
            individual = 1

        xmin = None
        xmax = None
        ymin = None
        ymax = None
        individuals = None
        (tracks, timesteps, individuals, xmin, xmax, ymin, ymax, zmin, zmax) = self.Setup_plot_boundaries(tracks, timesteps, individuals,\
                                                 xmin, xmax, ymin, ymax, zmin, zmax)

        if(not(individual in tracks.Individual.unique().tolist())):
            raise ValueError("Individual " + str(individual) + " not in dataset.")

        #set the stage limits if not yet set
        if(stagemin == None):
            stagemin = tracks.istage.min()
        if(stagemax == None):
            stagemax = tracks.istage.max()

        #read required data
        individual_tracks = tracks.loc[(tracks.Individual == individual) &\
                     (tracks.xa > -1000.0) & (tracks.ya > -1000.0) &\
                     (tracks.istage >= stagemin) & (tracks.istage <= stagemax),]

        time      = ((individual_tracks.Timestep - individual_tracks.Timestep.min()) * timestep) / 3600
        tempavr   = individual_tracks.dtemp
        vzact     = individual_tracks.vzact
        wsettl    = (individual_tracks.zdepth - individual_tracks.zdepth.shift(1)) / timestep
        wsettl[0] = 0
        wsettl    = wsettl[0:len(time)]
        stage     = individual_tracks.istage

        #set the time limits if not yet set
        if(timemin == None):
            timemin = np.nanmin(individual_tracks.Timestep - individual_tracks.Timestep.min())
        if(timemax == None):
            timemax = np.nanmax(individual_tracks.Timestep - individual_tracks.Timestep.min())

        fig = fig, axs = plt.subplots(4, 1, sharex=True)
        fig.subplots_adjust(hspace=0.1)

        axs[0].plot(time,tempavr,color='navy',linewidth=1)
        axs[0].set_ylabel('Temperature (''$^o$''C)',fontsize=6,weight="bold")
        
        #min_settl = min(min(vzact),min(wsettl))
        #max_settl = max(max(vzact),max(wsettl))
        axs[1].plot(time,vzact,color='lime',linewidth=1)
        axs[1].set_ylabel('Part. sett.\n vel. (m/s)',fontsize=6,weight="bold")
        #axs[1].set_ylim(min_settl,max_settl)
        axs[1].set_xlim((timemin * timestep) / 3600, (timemax * timestep) / 3600)
        
        axs[2].plot(time,wsettl,color='lime',linewidth=1)
        axs[2].set_ylabel('Sett. vel. \n vert. eddy diff. (m/s)',fontsize=6,weight="bold")
        #axs[2].set_ylim(min_settl,max_settl)
        axs[2].set_xlim((timemin * timestep) / 3600, (timemax * timestep) / 3600)

        axs[3].plot(time,stage)
        axs[3].set_ylabel('Stage (nr)',fontsize=6,weight="bold")
        axs[3].set_xlim((timemin * timestep) / 3600, (timemax * timestep) / 3600)
        axs[3].yaxis.set_major_locator(MaxNLocator(integer=True))

        axs[3].grid(alpha=0.5)
        axs[2].grid(alpha=0.5)
        axs[1].grid(alpha=0.5)
        axs[0].grid(alpha=0.5)
        
        plt.xlabel('Time (hr)',fontsize=13,weight="bold")
        axs[0].set_title('Overview : individual ' + str(individual))
        fig.align_ylabels()
        plt.show()

        return()


    def Static_particle_vs_time_verticle(self, tracks = None, timesteps = None, individual = None, timemin = None, \
        timemax = None, zmin = None, zmax = None, stagemin = None, stagemax = None):

        if(not(isinstance(individual,int))):
            raise ValueError("Individual should be one individual nr")

        if(individual == None):
            individual = 1

        xmin = None
        xmax = None
        ymin = None
        ymax = None
        individuals = None
        (tracks, timesteps, individuals, xmin, xmax, ymin, ymax, zmin, zmax) = self.Setup_plot_boundaries(tracks, timesteps, individuals,\
                                                 xmin, xmax, ymin, ymax, zmin, zmax)

        if(not(individual in tracks.Individual.unique().tolist())):
            raise ValueError("Individual " + str(individual) + " not in dataset.")

        #set the stage limits if not yet set
        if(stagemin == None):
            stagemin = tracks.istage.min()
        if(stagemax == None):
            stagemax = tracks.istage.max()

        #read required data
        individual_tracks = tracks.loc[(tracks.Individual == individual) &\
                     (tracks.xa > -1000.0) & (tracks.ya > -1000.0) &\
                     (tracks.istage >= stagemin) & (tracks.istage <= stagemax),]

        time      = (individual_tracks.Timestep - individual_tracks.Timestep.min()) 
        tempavr   = individual_tracks.dtemp
        partdepth = individual_tracks.zdepth * -1
        bottdepth = individual_tracks.zdepth * -1 + individual_tracks.zlevel * -1 
        #wsettl    = individual_tracks.wsettl
        stage     = individual_tracks.istage

        #set the time limits if not yet set
        if(timemin == None):
            timemin = np.nanmin(individual_tracks.Timestep - individual_tracks.Timestep.min())
        if(timemax == None):
            timemax = np.nanmax(individual_tracks.Timestep - individual_tracks.Timestep.min())



        fig = fig, axs = plt.subplots(4, 1, sharex=True)
        fig.subplots_adjust(hspace=0.1)

        axs[0].plot(time,tempavr,color='navy',linewidth=1)
        axs[0].set_ylabel('Temperature ''$^o$''C',fontsize=6,weight="bold")
        
        min_depth = min(min(partdepth),min(bottdepth))
        max_depth = max(max(partdepth),max(bottdepth))
        axs[1].plot(time,partdepth,color='lime',linewidth=1)
        axs[1].set_ylabel('Particle depth (m)',fontsize=6,weight="bold")
        axs[1].set_ylim(min_depth,max_depth)
        axs[1].set_xlim(timemin, timemax)
        
        axs[2].plot(time,bottdepth)
        axs[2].set_ylabel('Bathymetry (m)',fontsize=6,weight="bold")
        axs[2].set_ylim(min_depth,max_depth)
        axs[2].set_xlim(timemin, timemax)

        axs[3].plot(time,stage)
        axs[3].set_ylabel('Stage (nr)',fontsize=6,weight="bold")
        axs[3].set_xlim(timemin, timemax)
        axs[3].yaxis.set_major_locator(MaxNLocator(integer=True))

        axs[3].grid(alpha=0.5)
        axs[2].grid(alpha=0.5)
        axs[1].grid(alpha=0.5)
        axs[0].grid(alpha=0.5)
        
        plt.xlabel('Time',fontsize=13,weight="bold")
        axs[0].set_title('Overview : individual ' + str(individual))
        fig.align_ylabels()
        plt.show()

        return()


    def Static_particle_vs_distance_verticle(self, tracks = None, timesteps = None, individual = None, distmin = None, \
        distmax = None, zmin = None, zmax = None, stagemin = None, stagemax = None):

        if(not(isinstance(individual,int))):
            raise ValueError("Individual should be one individual nr")

        if(individual == None):
            individual = 1

        xmin = None
        xmax = None
        ymin = None
        ymax = None
        individuals = None
        (tracks, timesteps, individuals, xmin, xmax, ymin, ymax, zmin, zmax) = self.Setup_plot_boundaries(tracks, timesteps, individuals,\
                                                 xmin, xmax, ymin, ymax, zmin, zmax)

        if(not(individual in tracks.Individual.unique().tolist())):
            raise ValueError("Individual " + str(individual) + " not in dataset.")

        #set the stage limits if not yet set
        if(stagemin == None):
            stagemin = tracks.istage.min()
        if(stagemax == None):
            stagemax = tracks.istage.max()

        #read required data
        individual_tracks = tracks.loc[(tracks.Individual == individual) &\
                     (tracks.xa > -1000.0) & (tracks.ya > -1000.0),]

        #calculate distance using pythagoras
        individual_tracks = individual_tracks.sort_values(by=['Individual','Timestep'])
        individual_tracks["prev_xa"] = individual_tracks.xa.shift(1).where(\
            individual_tracks.Individual.eq(individual_tracks.Individual.shift(1)))
        individual_tracks["prev_ya"] = individual_tracks.ya.shift(1).where(\
            individual_tracks.Individual.eq(individual_tracks.Individual.shift(1)))
        individual_tracks["distance"] = individual_tracks.apply(lambda x : np.sqrt(abs(x.xa - x.prev_xa)**2 +\
                                         abs(x.ya - x.prev_ya)**2), axis = 1)
        individual_tracks["distance_cum"] = individual_tracks.groupby(['Individual'])["distance"].cumsum()

        #select stages shown
        selected_tracks = individual_tracks.loc[(individual_tracks.istage >= stagemin) &\
                                               (individual_tracks.istage <= stagemax),]

        #time      = (selected_tracks.Timestep - selected_tracks.Timestep.min()) 
        #xa        = selected_tracks.xa
        #ya        = selected_tracks.ya
        tempavr   = selected_tracks.dtemp
        partdepth = selected_tracks.zdepth * -1
        bottdepth = selected_tracks.zdepth * -1 + selected_tracks.zlevel * -1 
        #wsettl    = selected_tracks.wsettl
        stage     = selected_tracks.istage
        distance  = selected_tracks.distance_cum

        #set the distance limits if not yet set
        if(distmin == None):
            distmin = distance.min()
        if(distmax == None):
            distmax = distance.max()
        
        fig = fig, axs = plt.subplots(4, 1, sharex=True)
        fig.subplots_adjust(hspace=0.1)

        axs[0].plot(distance,tempavr,color='navy',linewidth=1)
        axs[0].set_ylabel('Temperature ''$^o$''C',fontsize=6,weight="bold")
        
        min_depth = min(min(partdepth),min(bottdepth))
        max_depth = max(max(partdepth),max(bottdepth))
        axs[1].plot(distance,partdepth,color='lime',linewidth=1)
        axs[1].set_ylabel('Particle depth (m)',fontsize=6,weight="bold")
        axs[1].set_ylim(min_depth,max_depth)
        axs[1].set_xlim(distmin, distmax)
        
        axs[2].plot(distance,bottdepth)
        axs[2].set_ylabel('Bathymetry (m)',fontsize=6,weight="bold")
        axs[2].set_ylim(min_depth,max_depth)
        axs[2].set_xlim(distmin, distmax)

        axs[3].plot(distance,stage)
        axs[3].set_ylabel('Stage (nr)',fontsize=6,weight="bold")
        axs[3].set_xlim(distmin, distmax)
        axs[3].yaxis.set_major_locator(MaxNLocator(integer=True))

        axs[3].grid(alpha=0.5)
        axs[2].grid(alpha=0.5)
        axs[1].grid(alpha=0.5)
        axs[0].grid(alpha=0.5)
        
        plt.xlabel('Distance',fontsize=13,weight="bold")
        axs[0].set_title('Overview : individual ' + str(individual))
        fig.align_ylabels()
        plt.show()

        return()

    def Static_particle_distance_histogram(self, tracks = None, timesteps = None, individuals = None, xmin = None, \
        xmax = None, ymin = None, ymax = None, stagenr = None, position = "first", distmin = None, distmax = None, binwidth = 100):
        
        backgroundmap = False

        zmin = None
        zmax = None

        #Check position
        if(not(position == "first" or position == "last")):
            raise ValueError("'position' should be 'last' or 'first'")

        #if(bndmodelshp == None):
        #    raise ValueError ("a model shapeboundary is needed for this plot")

        #load model boundery
        #model_boundary = geopandas.read_file(bndmodelshp, crs={'init' : 'epsg:4326'})
                
        if(stagenr == None):
            stagenr = 1
       
        (tracks, timesteps, individuals, xmin, xmax, ymin, ymax, zmin, zmax) = self.Setup_plot_boundaries(tracks, timesteps, individuals,\
                                                 xmin, xmax, ymin, ymax, zmin, zmax)

        individual_tracks = tracks.loc[(tracks.Individual.isin(individuals)) &\
                          (tracks.xa > -1000.0) & (tracks.ya > -1000.0) &\
                          (tracks.xa != 0.0) & (tracks.ya != 0.0),]

        #calculate distance using pythagoras
        individual_tracks = individual_tracks.sort_values(by=['Individual','Timestep'])
        individual_tracks["prev_xa"] = individual_tracks.xa.shift(1).where(\
            individual_tracks.Individual.eq(individual_tracks.Individual.shift(1)))
        individual_tracks["prev_ya"] = individual_tracks.ya.shift(1).where(\
            individual_tracks.Individual.eq(individual_tracks.Individual.shift(1)))
        individual_tracks["distance"] = individual_tracks.apply(lambda x : np.sqrt(abs(x.xa - x.prev_xa)**2 +\
                                         abs(x.ya - x.prev_ya)**2), axis = 1)
        individual_tracks["distance_cum"] = individual_tracks.groupby(['Individual'])["distance"].cumsum()

        #set the distance limits if not yet set
        if(distmin == None):
            distmin = individual_tracks["distance_cum"].min()
        if(distmax == None):
            distmax = individual_tracks["distance_cum"].max()

        #select first or last position of stage
        if(position == "first"):
            selected_positions = individual_tracks.loc[(individual_tracks.istage == stagenr),]
            selected_positions = selected_positions.groupby(['Individual','istage']).head(1).reset_index()
            selected_positions = selected_positions.loc[((selected_positions.distance_cum >= distmin) &\
        				(selected_positions.distance_cum <= distmax)),]
        else:
            selected_positions = individual_tracks.loc[(individual_tracks.istage == stagenr),]
            selected_positions = selected_positions.groupby(['Individual','istage']).tail(1).reset_index()
            selected_positions = selected_positions.loc[((selected_positions.distance_cum >= distmin) &\
        				(selected_positions.distance_cum <= distmax)),]


        #time      = (selected_positions.Timestep - selected_positions.Timestep.min()) 
        #xa        = selected_positions.xa
        #ya        = selected_positions.ya
        #tempavr   = selected_positions.dtemp
        #partdepth = selected_positions.zdepth * -1
        #bottdepth = selected_positions.zdepth * -1 + selected_positions.zlevel * -1
        #wsettl    = selected_positions.wsettl
        #stage     = selected_positions.istage
        distance  = selected_positions.distance_cum

        fig = fig, axs = plt.subplots(1, 1, sharex=True)
        fig.subplots_adjust(hspace=0.1)
                
        #Make histogram
        axs.hist(distance, bins = max(1,int((distmax - distmin) / binwidth)))
        axs.set_ylabel('Nr of particles (nr)',fontsize=6,weight="bold")
        axs.set_xlim(distmin, distmax)
        axs.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        #finish plot
        plt.xlabel('Distance',fontsize=13,weight="bold")
        plt.title('Overview : distance histogram',weight='bold')
        plt.show()

        return()


    def Static_particle_kernel_horizontal(self, tracks = None, timesteps = None, individuals = None, xmin = None, \
        xmax = None, ymin = None, ymax = None, stagenr = None, position = "first"):
        
        backgroundmap = False

        zmin = None
        zmax = None

        #Check position
        if(not(position == "first" or position == "last")):
            raise ValueError("'position' should be 'last' or 'first'")

        #if(bndmodelshp == None):
        #    raise ValueError ("a model shapeboundary is needed for this plot")

        #load model boundery
        #model_boundary = geopandas.read_file(bndmodelshp, crs={'init' : 'epsg:4326'})
                
        if(stagenr == None):
            stagenr = 1
       
        (tracks, timesteps, individuals, xmin, xmax, ymin, ymax, zmin, zmax) = self.Setup_plot_boundaries(tracks, timesteps, individuals,\
                                                 xmin, xmax, ymin, ymax, zmin, zmax)
        #TODO DEVELOP HORIZONTAL OVERVIEW
        #if(self.xmin > 100000 and self.xmin < 999999):
        (latmin, lonmin) = utm.to_latlon(xmin, ymin, 16, 'N')
        (latmax, lonmax) = utm.to_latlon(xmax, ymax, 16, 'N')
        backgroundmap = True
        

        fig = fig, axs = plt.subplots(1, 1, sharex=True)
        fig.subplots_adjust(hspace=0.1)

        individual_tracks = tracks.loc[(tracks.Individual.isin(individuals)) &\
                          (tracks.xa > -1000.0) & (tracks.ya > -1000.0) &\
                          (tracks.xa != 0.0) & (tracks.ya != 0.0) &
                          (tracks.istage == stagenr),]
        if(position == "first"):
            selected_positions = individual_tracks.groupby(['Individual','istage']).head(1).reset_index()
        else:
            selected_positions = individual_tracks.groupby(['Individual','istage']).tail(1).reset_index()

        #time      = (selected_positions.Timestep - selected_positions.Timestep.min()) 
        xa        = selected_positions.xa
        ya        = selected_positions.ya
        #tempavr   = selected_positions.dtemp
        #partdepth = selected_positions.zdepth * -1
        #bottdepth = selected_positions.zdepth * -1 + selected_positions.zlevel * -1
        #wsettl    = selected_positions.wsettl
        #stage     = selected_positions.istage

        if(backgroundmap):
            map=Basemap(llcrnrlon=lonmin,llcrnrlat=latmin, urcrnrlon=lonmax, urcrnrlat=latmax, resolution='h', projection='merc')
            map.fillcontinents(color='tan',lake_color='#DDEEFF',zorder=1)
            map.shadedrelief(alpha=0.5,zorder=0)
            map.drawrivers(color='#DDEEFF')
            
            map.drawmeridians(np.arange(np.floor(map.lonmin),np.ceil(map.lonmax),0.2),labels=[0,0,0,1],linewidth=0.8,dashes=[1,2])
            map.drawparallels(np.arange(np.floor(map.latmin),np.ceil(map.latmax),0.1),labels=[0,1,1,0],linewidth=0.8,dashes=[1,2])
            latlon   = [utm.to_latlon(row[0], row[1], 16, 'N') for row in zip(xa,ya)]
            Lat_tracks = [coord[0] for coord in latlon]
            Lon_tracks = [coord[1] for coord in latlon]
            x, y = map(Lon_tracks, Lat_tracks)

            map.scatter(x,y,marker='o',color='blue',s=10,linewidth=0.3,edgecolor='k',label='Position',zorder=10)
            # gdf = geopandas.GeoDataFrame(
            #         selected_positions, crs={'init' : 'epsg:4326'}, geometry=geopandas.points_from_xy(xa, ya)) 

            # #Show kernels
            # cbar_kws = {'extend':'both', 
            #             'extendfrac':None, 
            #             'drawedges':True, 
            #             'ticks': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], # set ticks of color bar
            #             'label':'Color Bar'}
            
            # axs = geoplot.kdeplot(gdf, vmin=0, vmax=1000, n_levels = 10,  cmap='Reds', cbar = True, cbar_kws = cbar_kws,  clip=model_boundary.geometry,  shade=True, shade_lowest=False,
            #         projection=geoplot.crs.AlbersEqualArea(), extent = model_boundary.total_bounds, ax=axs)
                
        else:
            x = xa.tolist()
            y = ya.tolist()
            
            map.scatter(x,y,marker='o',color='blue',s=10,linewidth=0.3,edgecolor='k',label='Position',zorder=10)
            
            # gdf = geopandas.GeoDataFrame(
            #         selected_positions, crs={'init' : 'epsg:4326'}, geometry=geopandas.points_from_xy(xa, ya)) 

            # #Show kernels
            # cbar_kws = {'extend':'both', 
            #             'extendfrac':None, 
            #             'drawedges':True, 
            #             'ticks': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], # set ticks of color bar
            #             'label':'Color Bar'}
            
            # geoplot.kdeplot(gdf, vmin=0, vmax=1000, n_levels = 10,  cmap='Reds', cbar = True, cbar_kws = cbar_kws,  clip=model_boundary.geometry,  shade=True, shade_lowest=False,
            #         projection=geoplot.crs.AlbersEqualArea(), extent = model_boundary.total_bounds, ax=axs)
                
        
        plt.title('Overview : kernals',weight='bold')
        plt.show()

        return()


    def Static_particle_vs_time_horizontal(self, tracks = None, timesteps = None, individuals = None, xmin = None, \
        xmax = None, ymin = None, ymax = None):
        
        backgroundmap = False

        if(individuals == None):
            individuals = [1]

        zmin = None
        zmax = None
       
        (tracks, timesteps, individuals, xmin, xmax, ymin, ymax, zmin, zmax) = self.Setup_plot_boundaries(tracks, timesteps, individuals,\
                                                 xmin, xmax, ymin, ymax, zmin, zmax)

        #TODO DEVELOP HORIZONTAL OVERVIEW
        if(xmin > 100000 and xmin < 999999):
            (latmin, lonmin) = utm.to_latlon(xmin, ymin, 16, 'N')
            (latmax, lonmax) = utm.to_latlon(xmax, ymax, 16, 'N')
            backgroundmap = True

        fig = fig, axs = plt.subplots(1, 1, sharex=True)
        fig.subplots_adjust(hspace=0.1)

        for individual in individuals:
            #read required data
            if(not(individual in tracks.Individual.unique().tolist())):
                raise ValueError("Individual " + str(individual) + " not in dataset.")

            individual_tracks = tracks.loc[(tracks.Individual == individual) &\
                         (tracks.xa > -1000.0) & (tracks.ya > -1000.0) &\
                          (tracks.xa != 0.0) & (tracks.ya != 0.0),]

            #time      = (individual_tracks.Timestep - individual_tracks.Timestep.min()) 
            xa        = individual_tracks.xa
            ya        = individual_tracks.ya
            #tempavr   = individual_tracks.dtemp
            #partdepth = individual_tracks.zdepth * -1
            #bottdepth = individual_tracks.zdepth * -1 + individual_tracks.zlevel * -1
            #wsettl    = individual_tracks.wsettl
            #stage     = individual_tracks.istage

            if(backgroundmap):
                map=Basemap(llcrnrlon=lonmin,llcrnrlat=latmin, urcrnrlon=lonmax, urcrnrlat=latmax, resolution='h', projection='merc')
                map.fillcontinents(color='tan',lake_color='#DDEEFF',zorder=1)
                map.shadedrelief(alpha=0.5,zorder=0)
                map.drawrivers(color='#DDEEFF')
            
                map.drawmeridians(np.arange(np.floor(map.lonmin),np.ceil(map.lonmax),0.2),labels=[0,0,0,1],linewidth=0.8,dashes=[1,2])
                map.drawparallels(np.arange(np.floor(map.latmin),np.ceil(map.latmax),0.1),labels=[0,1,1,0],linewidth=0.8,dashes=[1,2])
                latlon   = [utm.to_latlon(row[0], row[1], 16, 'N') for row in zip(xa,ya)]
                Lat_tracks = [coord[0] for coord in latlon]
                Lon_tracks = [coord[1] for coord in latlon]
                x, y = map(Lon_tracks, Lat_tracks)

                map.plot(x,y,color='gray',alpha=0.8,linewidth=0.5, zorder = 10)
                map.scatter(x[0],y[0],marker='o',color='green',s=30,linewidth=0.3,edgecolor='k',label='Initial',zorder=10)
                map.scatter(x[-1],y[-1],marker='o',color='blue',s=30,linewidth=0.3,edgecolor='k',label='Active',zorder=10)
            
            else:
                x = xa.tolist()
                y = ya.tolist()
                
                axs.plot(x,y,color='gray',alpha=0.8,linewidth=0.5, zorder = 10)
                axs.scatter(x[0],y[0],marker='o',color='green',s=30,linewidth=0.3,edgecolor='k',label='Initial',zorder=10)
                axs.scatter(x[-1],y[-1],marker='o',color='blue',s=30,linewidth=0.3,edgecolor='k',label='Active',zorder=10)
             
                

            plt.text(x[0],y[0],str(individual),color='midnightblue')
            plt.text(x[-1],y[-1],str(individual),color='midnightblue')

        plt.title('Overview : individuals ' + str(individuals),weight='bold')
        plt.show()

        return()

    def Static_3D_visualisation_Trackline(self, tracks = None, timesteps = None, individuals = None, xmin = None, \
        xmax = None, ymin = None, ymax = None, zmin = None, zmax = None): 

        (tracks, timesteps, individuals, xmin, xmax, ymin, ymax, zmin, zmax) = self.Setup_plot_boundaries(tracks, timesteps, individuals,\
                                                 xmin, xmax, ymin, ymax, zmin, zmax)


        # Attaching 3D axis to the figure
        fig = plt.figure()
        ax = p3.Axes3D(fig)
       
        # Setting the axes properties
        ax.set_xlim3d([xmin, xmax])
        ax.set_xlabel('X')
        ax.set_ylim3d([ymin, ymax])
        ax.set_ylabel('Y')
        ax.set_zlim3d([zmin, zmax])
        ax.set_zlabel('Z')
        ax.set_title('3D Test')

        (fig, ax, data, lines) = self.Setup_visualize_TrackLine(fig, ax, tracks, timesteps,individuals, xmin,\
                                     xmax, ymin, ymax, zmin, zmax)

        plt.plot([i for i in range(len(data[0][2]))],data[0][2])
        plt.show()

        return()

    def Dynamic_3D_visualisation_Trackline(self, tracks = None, timesteps = None, individuals = None, xmin = None, \
        xmax = None, ymin = None, ymax = None, zmin = None, zmax = None):

        (tracks, timesteps, individuals, xmin, xmax, ymin, ymax, zmin, zmax) = self.Setup_plot_boundaries(tracks, timesteps, individuals,\
                                                 xmin, xmax, ymin, ymax, zmin, zmax)

        # Attaching 3D axis to the figure
        fig = plt.figure()
        ax = p3.Axes3D(fig)
       
        # Setting the axes properties
        ax.set_xlim3d([xmin, xmax])
        ax.set_xlabel('X')
        ax.set_ylim3d([ymin, ymax])
        ax.set_ylabel('Y')
        ax.set_zlim3d([zmin, zmax])
        ax.set_zlabel('Z')
        ax.set_title('3D Test')
        
        (fig, ax, data, lines) = self.Setup_visualize_TrackLine(fig, ax, tracks, timesteps,individuals, xmin,\
                                     xmax, ymin, ymax, zmin, zmax)

        # Creating the Animation object
        line_ani = animation.FuncAnimation(fig, self.update_lines, timesteps, fargs=(data, lines),
                              blit=False)

        plt.show()

        return()

# Fifty lines of random 3-D lines
#individuals = 1

if __name__ == '__main__':
    #all_tracks = read_tracks("d:/Laptop/Deltares/Projects/2018/IBM_modelling/run_new/run_part_flow_0.1ms_b6_diurnal/", columnnames = colnames)
    # partdata = PARTTracks("d:/Projects/IBM_test_model/z-part_ibm_test_zlayer/")
    # partdata = PARTTracks("p:/11205918-epm/05 ASIAN CARP/_model_scenarios_forward/4_grasscarp_egg_wheeler_release_20200613/")
    #partdata = PARTTracks("p:/11205918-epm/05 ASIAN CARP/_model_scenarios_forward/4b_grasscarp_egg_wheeler_release_20200616/")
    # partdata = PARTTracks("p:/11205918-epm/05 ASIAN CARP/_model_scenarios_forward/5_grasscarp_egg_wheeler_release_20200104/")
    # partdata = PARTTracks("d:/Projects/IBM_test_model/4_grasscarp_egg_wheeler_release_20200613/")
    # partdata = PARTTracks("p:/11205918-epm/05 ASIAN CARP/_model_scenarios_forward/6_grasscarp_egg_wheeler_release_20200606/")
    # partdata = PARTTracks("p:/11205918-epm/05 ASIAN CARP/_model_scenarios_forward/7_grasscarp_egg_wheeler_release_20200601/")
    # partdata = PARTTracks("p:/11205918-epm/05 ASIAN CARP/_model_scenarios_forward/8_grasscarp_egg_wheeler_release_inc_larvae_mod_20200104/")
    # partdata = PARTTracks("p:/11205918-epm/05 ASIAN CARP/_model_scenarios_forward/9_grasscarp_egg_wheeler_release_inc_larvae_mod_20200606/")
    # partdata = PARTTracks("p:/11205918-epm/05 ASIAN CARP/_model_scenarios_forward/10_grasscarp_egg_wheeler_release_inc_larvae_mod_20200601/")
    # partdata = PARTTracks("p:/11205918-epm/05 ASIAN CARP/_model_scenarios_forward/11_grasscarp_egg_wheeler_release_inc_larvae_mod20200601_5000particles/")
    # partdata = PARTTracks("p:/11205918-epm/05 ASIAN CARP/_model_scenarios_reversed/1_grasscarp_egg_wheeler_rev_release_inc_larvae_mod20200601_1particles/")
    # partdata = PARTTracks("p:/11205918-epm/05 ASIAN CARP/_model_scenarios_reversed/2_grasscarp_egg_wheeler_rev_release_inc_larvae_mod20200701_5000particles/")
    #partdata = PARTTracks("p:/11205918-epm/05 ASIAN CARP/_model_scenarios_reversed/3_grasscarp_egg_wheeler_rev_release_inc_larvae_mod20200616_5000particles/")
    #partdata = PARTTracks("p:/11205918-epm/05 ASIAN CARP/_model_scenarios_reversed/4_grasscarp_egg_wheeler_rev_release_inc_larvae_mod_20200607_5000particles/")
    #partdata = PARTTracks("p:/11205918-epm/05 ASIAN CARP/_model_scenarios_reversed/5_grasscarp_egg_wheeler_rev_release_inc_larvae_mod_20200603_5000particles/")
    # partdata = PARTTracks("p:/11205918-epm/05 ASIAN CARP/_model_scenarios_forward/12_grasscarp_egg_wheeler_release_inc_larvae_mod20200601_5000particles_test_aftercrhonrev/")
    partdata = PARTTracks("p:/11205918-epm/05 ASIAN CARP/_model_scenarios_forward/12_grasscarp_egg_wheeler_release_inc_larvae_mod20200601_5000particles_test_aftercrhonrev/")
    # partdata = PARTTracks("p:/11205918-epm/05 ASIAN CARP/_model_scenarios_forward/13_grasscarp_egg_wheeler_release_inc_larvae_mod_20200104/")
    
    # partdata.Dynamic_3D_visualisation_Trackline(individuals = [1,50,80])
    # partdata.Static_particle_vs_time_verticle(individual = 10)
    # partdata.Static_particle_vs_time_verticle(individual = 20)
    # partdata.Static_particle_vs_time_verticle(individual = 30)
    # partdata.Static_particle_vs_time_verticle(individual = 40)
    # partdata.Static_particle_vs_time_verticle(individual = 50)
    
    #partdata.Static_particle_vs_time_horizontal(individuals = [10,20,30,40,50], xmin = 455000, xmax = 554352 , ymin = 3810000 , ymax = 3851000) 
    #partdata.Static_particle_vs_time_horizontal(individuals = [1000,2000,3000,4000,5000], xmin = 455000, xmax = 554352 , ymin = 3810000 , ymax = 3851000) 
    #partdata.Static_particle_vs_distance_verticle(individual = 10, distmax = 160000, stagemax = 2)
    #partdata.Static_particle_vs_distance_verticle(individual = 20, distmax = 160000, stagemax = 2)
    #partdata.Static_particle_vs_distance_verticle(individual = 30, distmax = 160000, stagemax = 2)
    #partdata.Static_particle_vs_distance_verticle(individual = 40, distmax = 160000, stagemax = 2)
    #partdata.Static_particle_vs_distance_verticle(individual = 50, distmax = 160000, stagemax = 2)
    #partdata.Static_particle_vs_distance_verticle(individual = 1000, distmax = 160000, stagemax = 2)
    #partdata.Static_particle_vs_distance_verticle(individual = 2000, distmax = 160000, stagemax = 2)
    #partdata.Static_particle_vs_distance_verticle(individual = 3000, distmax = 160000, stagemax = 2)
    #partdata.Static_particle_vs_distance_verticle(individual = 4000, distmax = 160000, stagemax = 2)
    #partdata.Static_particle_vs_distance_verticle(individual = 5000, distmax = 160000, stagemax = 2)
    #partdata.Static_particle_characteristics_vs_time_verticle(individual = 10, timestep = 300, stagemax = 1)
    #partdata.Static_particle_characteristics_vs_time_verticle(individual = 20, timestep = 300, stagemax = 1)
    #partdata.Static_particle_characteristics_vs_time_verticle(individual = 30, timestep = 300, stagemax = 1)
    #partdata.Static_particle_characteristics_vs_time_verticle(individual = 40, timestep = 300, stagemax = 1)
    #partdata.Static_particle_characteristics_vs_time_verticle(individual = 50, timestep = 300, stagemax = 1)
    #partdata.Static_particle_characteristics_vs_time_verticle(individual = 1000, timestep = 300, stagemin = 1, stagemax = 1)
    #partdata.Static_particle_characteristics_vs_time_verticle(individual = 2000, timestep = 300, stagemin = 1, stagemax = 1)
    #partdata.Static_particle_characteristics_vs_time_verticle(individual = 3000, timestep = 300, stagemin = 1, stagemax = 1)
    #partdata.Static_particle_characteristics_vs_time_verticle(individual = 4000, timestep = 300, stagemin = 1, stagemax = 1)
    #partdata.Static_particle_characteristics_vs_time_verticle(individual = 5000, timestep = 300, stagemin = 1, stagemax = 1)
    
    partdata.Static_particle_distance_histogram(xmin = 455000, xmax = 554352 , ymin = 3810000 , ymax = 3851000, stagenr = 3,\
     position = "first", distmin = 0, distmax = 160000, binwidth = 5000)
    #partdata.Static_particle_distance_histogram(xmin = 455000, xmax = 554352 , ymin = 3810000 , ymax = 3851000, stagenr = 1,\
    # position = "last", distmin = 0, distmax = 160000, binwidth = 5000)
    #partdata.Static_particle_kernel_horizontal(xmin = 455000, xmax = 554352 , ymin = 3810000 , ymax = 3851000, stagenr = 2,\
    # position = "last")
    #partdata.Static_particle_kernel_horizontal(xmin = 455000, xmax = 554352 , ymin = 3810000 , ymax = 3851000, stagenr = 1,\
    # position = "last")
    print("Done")

