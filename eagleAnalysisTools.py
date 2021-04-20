import eagleSqlTools as sql
import numpy as np
import matplotlib.pyplot as plt
from astroquery.sdss import SDSS
import lmfit as lm
import astropy
import time
import datetime
import pickle


class eagleData():
    def __init__(self, randomNumber, simulation, dataStorageLocation):
        '''
        The initialisation of this class initially will define all the constants that will be required during the module. These can be changed directly and then run the                      UpdateInformation module to update all the dependants. It also creates the dictionaries that will be used to store data during the processing. This initialisation requires           the parameters:
        - randomNumber: A random number used to limit the number of results queried from the database
        - simulation: The simulation that the we are interested in querying
        - dataStorageLocation: The location that the pickle files will be loaded from and saved to
        '''

        #Defining Values
        self.ageUniverse = 13.799 # Age of the Universe used in Time Linking estimate gotten from
        self.massPrecision = 0.05 # Mass Precision Used in Galaxy Matching
        self.dataStoragePath = dataStorageLocation # The path to the storage for files that will be save and loaded
        self.MinimumNumberInBin = 10 # The minimum number of values that need to be in a bin in order for it to be statistically signicant enough to take the median


        #Defining all the Storage Tables
        self.dataStorage = {}  # Main Raw Data Storage
        self.binningStorage = {} # Post Processing Storage for Binned Data
        self.comparisonData = {} # Storage of Galaxies that have been matched
        self.propertiesQueried = {} # Storage of the properties stored for each DataSet
        self.defaultQueries = {} # Storage of Default Queries
        self.timeData = astropy.table.Table() # Storage for the Time Data (relation between redshift, snapshot numbers, lookbacktime and age of the universe)
        self.timeReference = {} # A Reference Dictionary used to get corresponding redshfit and lookback time for a given redshift


        # DataBase Connection
        self.con = sql.connect("scp763", password="ih39ztr2")


        # PreInitialisation of Stored Queries
        self.simName = simulation
        self.randNum = randomNumber
        self.generateQueries()

        # Query Time Data from the Eagle Database to be used by the time linking module
        # Get the data from the Eagle Database
        snapShotData = sql.execute_query(self.con, self.defaultQueries['TimeQuery'])
        #Add the queried data to the storage Astropy Table self.timeData
        for label in snapShotData.dtype.names:
            self.timeData[label] = snapShotData[label]
        del snapShotData, label

        # Also calculate the corresping age of the universe at each snapshot and add it as a column
        self.timeData['AgeOfUniverse'] = self.ageUniverse - self.timeData['LookBackTime']

        # Creating a dictionary used for referenceing the snap shot to the redshift and lookbacktime 
        for SN, LB, z, t in zip(self.timeData['SnapNum'], self.timeData['LookBackTime'], self.timeData['Z'], self.timeData['AgeOfUniverse'] ):
            self.timeReference[SN] = [LB,z,t]


    def generateQueries(self):
        # This module generates the default queries using the current simulation and random number. If you wish to change the simulation and query for these default queries change             self.simName and self.randNum to desired values and either rerun this module of the updateDependants module
        # Below we add the newly generated default queries to the self.defaultQueries dictionary

        # Time Data Query
        self.defaultQueries['TimeQuery'] = "SELECT \
                SN.SnapNum as SnapNum, \
                SN.Redshift as Z, \
                SN.LookbackTime as LookBackTime, \
                Round(SN.Redshift, 2) as refZ \
            FROM \
                Snapshots as SN"

        #Progenitor Galaxies
        self.defaultQueries['ProgenitorGalaxies'] = "SELECT \
                    ProgGal.SnapNum as SnapShot, \
                    ProgGal.GalaxyID as GalaxyID, \
                    ProgGal.TopLeafID as TopLeafID, \
                    ProgGal.Redshift as Z, \
                    ProgGal.StarFormationRate as SFR, \
                    ProgGal.MassType_DM as DMmass, \
                    ProgGal.MassType_Star as SM, \
                    ProgGal.BlackHoleMass as sgBHM, \
                    ProgGal.MassType_BH as pBHM, \
                    ProgGal.BlackHoleMassAccretionRate as iBMAR, \
                    ProgGal.Vmax as Vmax, \
                    ProgGal.RandomNumber as RandNum \
                FROM \
                    {}_SubHalo as RefGal, \
                    {}_SubHalo as ProgGal \
                WHERE \
                    RefGal.TopLeafID = ProgGal.TopLeafID AND \
                    RefGal.SnapNum = 28 AND \
                    RefGal.SubGroupNumber = 0 AND \
                    RefGal.RandomNumber < {} \
                ORDER BY \
                    ProgGal.TopLeafID, \
                    ProgGal.Redshift".format(self.simName, self.simName, self.randNum)
        
        #Reference Galaxies
        self.defaultQueries['ReferenceGalaxies'] = "SELECT \
                SH.MassType_DM as DMmass, \
                SH.MassType_Star as SM, \
                SH.BlackHoleMassAccretionRate as BMAR, \
                SH.BlackHoleMass as sgBHM, \
                SH.MassType_BH as pBHM, \
                SH.StarFormationRate as SFR, \
                (Mag.g_nodust - Mag.r_nodust) as GR, \
                (Mag.r_nodust) as R, \
                (Mag.g_nodust) as G \
            FROM \
                {}_SubHalo as SH, \
                {}_Magnitudes as Mag \
            WHERE \
                SH.GalaxyID = Mag.GalaxyID AND \
                SH.SnapNum = 28 AND \
                SH.SubGroupNumber = 0 AND \
                SH.RandomNumber < {}".format(self.simName, self.simName, self.randNum)

        #Comparison Query
        self.defaultQueries['ComparisonGalaxies'] = "SELECT \
                fof.GroupCentreOfPotential_x as x, \
                fof.GroupCentreOfPotential_y as y, \
                fof.GroupCentreOfPotential_z as z, \
                fof.GroupMass as Mass, \
                fof.Group_R_Mean200 as radius, \
                fof.GroupID as GroupID, \
                SH.Vmax as Vmax, \
                SH.MassType_DM as DMmass, \
                SH.MassType_Star as SM, \
                SH.BlackHoleMassAccretionRate as BMAR, \
                SH.BlackHoleMass as sgBHM, \
                SH.MassType_BH as pBHM, \
                SH.StarFormationRate as SFR, \
                SH.SnapNum as SnapShot, \
                SH.GalaxyID as GalaxyID \
            FROM \
                {}_FOF as fof, \
                {}_Subhalo as SH \
            WHERE \
                SH.SubGroupNumber = 0 AND \
                SH.GroupID = fof.GroupID AND \
                fof.RandomNumber < {}".format(self.simName, self.simName, self.randNum)

        #Test Query
        self.defaultQueries['TestGalaxies'] = "SELECT \
                fof.GroupCentreOfPotential_x as x, \
                fof.GroupCentreOfPotential_y as y, \
                fof.GroupCentreOfPotential_z as z, \
                fof.GroupMass as Mass, \
                fof.Group_R_Mean200 as radius, \
                fof.GroupID as GroupID, \
                SH.Vmax as Vmax, \
                SH.StarFormationRate as SFR \
            FROM \
                {}_FOF as fof, \
                {}_Subhalo as SH \
            WHERE \
                SH.GroupID = fof.GroupID AND \
                fof.RandomNumber < {} \
            ORDER BY \
                fof.GroupMass DESC".format(self.simName, self.simName, self.randNum)


    def linkTimes(self, dataStorageTitle):
        '''
        The goal of this module is to add columns for the lookback time and age of the universe for all of the galaxies in the seelcted data source. This module requires the                 following parameters:
        - dataStorageTitle: This is the parameter label used with in the self.dataStorage dictionary to label the source of Data
        '''
        # The first step is checking whether the selected data already has the time data:
        if self.propertiesQueried[dataStorageTitle].count('LookBackTime') == 1 or self.propertiesQueried[dataStorageTitle].count('AgeOfUniverse') == 1 or self.propertiesQueried[dataStorageTitle].count('RedShift') == 1:
            print('ERROR \n One or both of the time parameters are present on the dataset already \n Aborting Function')
            return
        # This is to check whether there is snapshot information present in the data which is required to link the rest of the tie data
        if self.propertiesQueried[dataStorageTitle].count('SnapShot') != 1:
            print('Error \n There is no redshift present in the data to link time data to. Please try again with data with Redshit \n Aborting Function')
            return
        

        # Debug Timer Setup
        print('Adding Time Data to {} ...'.format(dataStorageTitle))
#         ts = time.time()

        # This part loops through every entry in the selected data table in order to link the appropriate time value in a couple of temp arrays later this is then added as addition            columns in the orgininal datastorage
        lookBackTemp = []
        ageOfUniverseTemp = []
        redshiftTemp = []
        for SN in self.dataStorage[dataStorageTitle]['SnapShot']:
            time = self.timeReference[SN]
            lookBackTemp.append(time[0])
            redshiftTemp.append(time[1])
            ageOfUniverseTemp.append(time[2])

        # Saving the calculated time data to the appropriate dataset
        self.dataStorage[dataStorageTitle]['LookBackTime'] = lookBackTemp
        self.dataStorage[dataStorageTitle]['AgeOfUniverse'] =  ageOfUniverseTemp
        self.dataStorage[dataStorageTitle]['RedShift'] =  redshiftTemp

#         del lookBackTemp, ageOfUniverseTemp, redshiftTemp, time
        
        # Debug Timer Finalising
#         calcTime = time.time() - ts
#         t = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
#         log = '{}   |  Sort Time: {}s'.format(t,round(calcTime,2))
#         print(log)

#         del  ts, calcTime, t, log


    
    def saveData(self, datasetTitle):
        '''
        The purpose of this function is to save a dataset to pickle file so that it can be loaded at a later date to save time in the data reduction processs. It requires the                following parameters:
        - datasetTitle: The lable used in the storage of data within the dataStorage dictionary, the file saved will also be saved in the format of "dataStoragePath/datasetTitle.p"
        '''
        #Code to dump a dataset in the datastorageLocation
        pickle.dump(self.dataStorage[datasetTitle], open("{}{}.p".format(self.dataStoragePath,datasetTitle),"wb") )





    def DatabaseQuery(self, title, queryOrQueryTitle, defaultQuery = False, loadFromPickle = False, convertToAstropyOnLoad = True):
        ''' 
        The aim of this module will be to query the EAGLE database, import this data into the python module and sort it for easy access for automation but also allowing for                  intuitive manual access. This moduel takes the following parameters:
        
        - title: A title for the Data that is going to be queried (this will be the label of data in the storage dictionary)
        - Query: This should either be a fully formed SQL query startiing with the word SELECT with no spaces beforehand or if the parameter defaultQuery is set to True a string               stating the name of one of the defaul queries availble in the package.
        - defaultQuery: This parameter controls whether the module is expecting a custon SQL query or will be using a defualt query stored in the module
        - loadFromPickle: If this parameter is set to true the module will instead load the data from a previously saved pickle file and organise it as required 
        - convertToAstropyOnLoad: If this parameter is true the imported pickle file will be converted to a Astropy table (mainly used to stop incosistant length dictionaries being            converted, as this would rasie and error)

        '''

        #Firstly we will test loadFromPickle to known whether there is a pickle file to be loaded or the database will need to be queried
        if loadFromPickle:
            # Firstly we need to check the data type of the imported data. This is only going to support data which is already in the astropy.table.table.Table format or in hte dict               type which would then need to be sorted into the astropy.table.table.Table
            # Loading the specified pickle file
            queriedData = pickle.load(open("{}{}.p".format(self.dataStoragePath,queryOrQueryTitle), "rb"))
            
            #Checking if the Pickle file is a astropy.table.table.Table type
            if isinstance(queriedData, astropy.table.table.Table):
                # Saving the astropy table to the central data storage
                self.dataStorage[title] = queriedData
                self.propertiesQueried[title] = queriedData.keys()

                del queriedData
            # Checking if the Pickle file is a dict type
            elif isinstance(queriedData, dict):
                if convertToAstropyOnLoad:
                    # We will continue to sort this data as if it were a query from the database with some debug timing
                    print('Imported Data File is in Diction Format \n Sorting Data into Astropy Table ...')
                    ts = time.time()
                    
                    #Creation Of Table
                    dataTable = astropy.table.Table()

                    for label in queriedData.keys():
                        dataTable[label] = queriedData[label]
                        
                    #Storing Data in Dictionary
                    self.dataStorage[title] = dataTable
                    self.propertiesQueried[title] = queriedData.dtypes.names

                    # Spare Variable Clean Up
                    del queriedData, dataTable, label

                    #Debug Timing Finalising
                    sortTime = time.time() - ts
                    t = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    log = '{}   |  Sort Time: {}s'.format(t,round(sortTime,2))
                    print(log)
                    del ts, sortTime, t , log

                else:
                    self.dataStorage[title] = queriedData
                    self.propertiesQueried[title] = queriedData.keys()

                    del queriedData
            
            #If it does not match the two supported formats return an error statement
            else:
                print("ERROR \n Chosen file does not match the supported formats. Please ensure the contents are either a Dictionary obeject or Astropy Table \n Aborting Function")


        
        else:
            # In this first section we will determine which query will be sent to the database by checking if a default query was selected
            if defaultQuery:
                # If a default query was selected we are grabbing a query from the dictionary stored in the class.
                 query = self.defaultQueries[queryOrQueryTitle]

            else:
                # If a default query was not selected we first test if the query has the appropriate start
                if queryOrQueryTitle[:6] != "SELECT":
                    print("ERROR in query Formatting: \n Please ensure all queries begin with the phrase SELECT and that all commands are formatted in uppercase \n ABORTING QUERY")
                    return
                else:
                    query = queryOrQueryTitle
                
            # Now the query for the array is selected the data base will be sent the query. There is additional code around the query to keep track of how long the query is taking                 which is useful for when large datasets are requested

            ####################
            # Debug Timing SetUp
            print('{} | Querying Database ... '.format(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
            ts = time.time()

            # Query the Database
            queriedData = sql.execute_query(self.con, query)

            # Debug Timing Finalising
            queryTime = time.time() - ts
            t = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log = '{}   |  Query Time: {}s'.format(t,round(queryTime,2))
            print(log)
            del ts, queryTime, t , log


            # Now the Database has been queried and the data has been stored in the variable queriedData, we are going to re sort this data to store it in a Astropy Table as this                  allows for better visualisation of the dataset when manually analysing the data
            # Firstly We Create the Astropy table and we also include a debuging timer to ensure time is not being lost when sorting the data

            ####################
            # Debug Timing SetUp
            print('Sorting Data into Astropy Table ...')
            ts = time.time()
            
            if convertToAstropyOnLoad:
                #Creation Of Table
                dataTable = astropy.table.Table()
                for label in queriedData.dtype.names:
                    dataTable[label] = queriedData[label]

                #Storing Data in Dictionary
                self.dataStorage[title] = dataTable
                self.propertiesQueried[title] = queriedData.dtype.names

                # Spare Variable Clean Up
                del queriedData, dataTable, label
            else: 
                self.dataStorage[title] = queriedData

            #Debug Timing Finalising
            sortTime = time.time() - ts
            t = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            log = '{}   |  Sort Time: {}s'.format(t,round(sortTime,2))
            print(log)
            del ts, sortTime, t , log



            # This is the end of the Data Aquisiation, the data is now stored a dictionary called dataStorage labeled with the title required as a parameter

    def binning(self, dataStorageTitle, numberOfBins, binVariable, variableToBeBinned, binningRange, loggedBins = False, loggedValues = False):
        '''
        The main purpose of this function will be to take data stored in self.dataStorage and bin the values in based on the column titles. If custom paramters for binning are               requried this should be done by adding a column to the dataset directly as this function will only work with column headings/lables. This function will require the following         parameters:
        - dataStorageTitle: This is the title used in the self.dataStorage dictionary to specify which dataset should be used
        - numberOfBins: This parameter specifies the number of bins that should be used
        - binVariable: This parameter specifies which variable in the dataStorage will be used to sepcifiy the size if the bins
        - variableToBeBinned: This parameter will be used to specify which variable in dataStorage will be sorted into the bins sepcified by binVariable
        - loggedBins: This paramter sepcifies whether the bins should be logarithmicly seperated.
        - binningRange: This parameter will be used to specify the upper and lower bounds of the binning. If loggedBins is true these binningRanges will be assumed to also be logged                         ranges
        '''
        # Testing Imput variables (This May be Implemented at a later Date)
       

        ######################
        # Firstly we will be creating temporary variables which will be used during the binning process
        binnedDataTemp = [] # Defining the temporary storage for the binned data
        binDataStorage = {} # Defining the dictionary to store all the binned data and averagin and will be saved to the main Storage after the processing

        #Next Gathering the values to be used in the binning process and taking their log if loggedBins was set to True
        if loggedBins:
            binValues = np.array( np.log10( self.dataStorage[dataStorageTitle][binVariable] ) )
        else:
            binValues = np.array( self.dataStorage[dataStorageTitle][binVariable] )
            
        if loggedValues:
            valuesToBeBinned = np.array( np.log10( self.dataStorage[dataStorageTitle][variableToBeBinned] ) )
        else:
            valuesToBeBinned = np.array( self.dataStorage[dataStorageTitle][variableToBeBinned] )

        #####################
        # This Section will create the binning range including the bin edges and the bin centers which will be used during the binning process
        #Firstly we calculate the width of the Bin given the binningRange and the number of Bins
        binWidth = abs((binningRange[1]) - (binningRange[0])) / numberOfBins
        binEdges = [(binningRange[0] + (i * binWidth)) for i in range(numberOfBins+1)]
        binCenters = [(binningRange[0] + (binWidth / 2)) + (i * binWidth) for i in range(numberOfBins)]
        

        #####################
        # This section carries out the bining process
        # We are going to loop of each bin here and create a mask to sort the data into the correct bin
        for i in range(len(binEdges)-1):
            # creating a mask between the ith bin edge and (i + 1)th bin edge for the binnnedVariable
            binMask = (binValues > binEdges[i]) & (binValues < binEdges[i+1]) 
           
            # appending all values which match the mask the temp binnedData storage
            binnedDataTemp.append( np.array(valuesToBeBinned[binMask]))
       

        # Testing to see if there are a significant enough number of galaxies in the bin to take the median. This value is set by self.MinimumNumberInBin
        canBeAveraged = np.array([len(x) for x in binnedDataTemp]) >= 10

        # Saving Binned Data to Storage
        binDataStorage['rawBinnedData'] = np.array(binnedDataTemp)
        binDataStorage['canBeAveraged'] = canBeAveraged
        binDataStorage['binCenters'] = np.array(binCenters)
        binDataStorage['binEdges'] = np.array(binEdges)

        del binnedDataTemp, canBeAveraged, binMask, binCenters, binEdges

        #####################
        # This section will carry out some averaging of the binned data
        # We will calculate the median, the 60-40 percentiles, 75-25 percentiles and the number density

        binDataStorage['Median'] = np.array( [np.median(x[~np.isnan(x)]) for x in binDataStorage['rawBinnedData']] )      
        binDataStorage['Mean'] = np.array( [np.mean(x[~np.isnan(x)]) for x in binDataStorage['rawBinnedData']] )  
        binDataStorage['Standard Error'] = np.array( [( np.std(x, ddof=1) / np.sqrt(np.size(x))) for x in binDataStorage['rawBinnedData']] )   
        binDataStorage['NumberDensity'] = np.array( [len(x) for x in binDataStorage['rawBinnedData']] )

        # These percentiles may need protection in place in case they fail but we shall wait and see
        binDataStorage['Percentile25'] = np.array( [np.percentile(x[~np.isnan(x)], 25) for x in binDataStorage['rawBinnedData']] ) 
        binDataStorage['Percentile40'] = np.array( [np.percentile(x[~np.isnan(x)], 40) for x in binDataStorage['rawBinnedData']] )
        binDataStorage['Percentile60'] = np.array( [np.percentile(x[~np.isnan(x)], 60) for x in binDataStorage['rawBinnedData']] )
        binDataStorage['Percentile75'] = np.array( [np.percentile(x[~np.isnan(x)], 75) for x in binDataStorage['rawBinnedData']] )

        ######################
        # Finally the temp dictionary will be stored in the main storage dictionary 
        self.binningStorage["{}:{} vs {}".format(dataStorageTitle, binVariable, variableToBeBinned)] = binDataStorage

        del binDataStorage


    def binningSnapShot(self, dataStorageTitle, binVariable, SNmin, SNmax):
        
        
        ######################
        # Firstly we will be creating temporary variables which will be used during the binning process
        binnedDataTemp = [] # Defining the temporary storage for the binned data
        binDataStorage = {} # Defining the dictionary to store all the binned data and averagin and will be saved to the main Storage after the processing
        binCenters = [] # Defining the temporary storage for the bin centers
        
        # Secondly creating a loop over all the snapshots
        SNrange = np.arange(SNmin, SNmax+1)
        for SN in SNrange:
            # Creating a mask for the current snapShot
            maskSN = self.dataStorage[dataStorageTitle]['SnapShot'] == SN
            binnedDataTemp.append(self.dataStorage[dataStorageTitle][binVariable][maskSN])
            binCenters.append(self.timeData['Z'][self.timeData['SnapNum'] == SN][0])
            
        # Testing to see if there are a significant enough number of galaxies in the bin to take the median. This value is set by self.MinimumNumberInBin
        canBeAveraged = np.array([len(x) for x in binnedDataTemp]) >= 10
        
        # Saving Binned Data to Storage
        binDataStorage['rawBinnedData'] = np.array(binnedDataTemp)
        binDataStorage['canBeAveraged'] = canBeAveraged
        binDataStorage['binCenters'] = np.array( binCenters )
        
        
        #####################
        # This section will carry out some averaging of the binned data
        # We will calculate the median, the 60-40 percentiles, 75-25 percentiles and the number density

        binDataStorage['Median'] = np.array( [np.median(x) for x in binDataStorage['rawBinnedData']] )      
        binDataStorage['Mean'] = np.array( [np.mean(x) for x in binDataStorage['rawBinnedData']] )  
        binDataStorage['Standard Error'] = np.array( [( np.std(x, ddof=1) / np.sqrt(np.size(x))) for x in binDataStorage['rawBinnedData']] )   
        binDataStorage['NumberDensity'] = np.array( [len(x) for x in binDataStorage['rawBinnedData']] )

        # These percentiles may need protection in place in case they fail but we shall wait and see
        binDataStorage['Percentile25'] = np.array( [np.percentile(x, 25) for x in binDataStorage['rawBinnedData']] ) 
        binDataStorage['Percentile40'] = np.array( [np.percentile(x, 40) for x in binDataStorage['rawBinnedData']] )
        binDataStorage['Percentile60'] = np.array( [np.percentile(x, 60) for x in binDataStorage['rawBinnedData']] )
        binDataStorage['Percentile75'] = np.array( [np.percentile(x, 75) for x in binDataStorage['rawBinnedData']] )

        ######################
        # Finally the temp dictionary will be stored in the main storage dictionary 
        self.binningStorage["{}:SnapShot vs {}".format(dataStorageTitle, binVariable)] = binDataStorage

        del binDataStorage
        
        
        
        
        
    
