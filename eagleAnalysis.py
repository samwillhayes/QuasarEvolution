# Modules Required for the Analysis
import numpy as np
import matplotlib.pyplot as plt
from astroquery.sdss import SDSS
import lmfit as lm
import astropy
import time
import datetime
import pickle
from scipy.optimize import curve_fit
import math
from scipy import stats
import uncertainties.unumpy as unp
import uncertainties as unc


class eagleReduction():
    def __init__(self, dataStorageLocation):
        '''The initialisation of this class will define all the constants that will be used during this module. 
        These can be changed directly and then run the UpdateInformation module to update all the dependants. 
        It also creates the dictionaries that will be used to store data during the processing. 
        This initialisation requires the parameters:
        - randomNumber: A random number used to limit the number of results queried from the database
        - simulation: The simulation that the we are interested in querying
        - dataStorageLocation: The location that the pickle files will be loaded from and saved to
        '''  
        
        ############################################
        #############Defining Constants#############
        ############################################
        
        self.ageUniverse = 13.799 # Age of the Universe used in Time Linking estimate gotten from
        self.massPrecision = 0.05 # Mass Precision Used in Galaxy Matching
        self.critVmax = 10**2.2 # Critical Virial Mass
        self.critTmax = 10**6 # Critical Virial Temperature
        
        # Physical Constants
        self.G = 6.67e-11
        
        # Model Constants
        self.kappa = 5./3
        self.epsilon = 0.2
        self.alpha = 1
        self.eta = 1.7 
        self.eF = 0.15
        self.eR = 0.1
        self.OmegaB = 0.15750563
        self.parsec = 3.086e16
        self.SolarMass = 1.99e30
        
        # Defining Paths and Strings
        self.dataStoragePath = dataStorageLocation # The path to the storage for files that will be save and loaded
        
        
        # Defining Storage Dictionaries
        self.Storage = {}
        self.binStorage = {}
        
        # Creating Dependants
        self.UpdateInformation()
        
    
    def UpdateInformation(self):
        '''This Function creates all of the dependant properties.
        The purpose is to update the dependant properties if an assumed constant is to be changed
        '''
        # Defining Energy Constants
        self.Vagn = ((2 * self.eR * self.eF) / (1 - self.eR))**(0.5) * (3e5)  # Units of kms^-1
        self.Vstar = 400 * ((self.epsilon / 0.091) * (self.eta / 1.74))**(0.5) # Units of kms^-1
        
        
    def LoadInData(self, dictionary, values):
        '''This function imports specific values from a dictionary and stores it in an internal dictionary. 
        Parameters passed through are:
        - dictionary: A diction type element 
        - values: name of keys within dictionary to be saved
        '''
        # Check Valid Keys
        for value in values:
            if value not in dictionary.keys():
                print('ERROR \n Invalid Key Value: {}'.format(value))
                return
        
        # Import Values
        for value in values:
            self.Storage[value] = np.array(dictionary[value])
                
    def Binning(self, numberOfBins, binVariable, variableToBeBinned, binningRange, loggedBins = False, loggedValues = False):
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
            binValues = np.array( np.log10( self.Storage[binVariable] ) )
        else:
            binValues = np.array( self.Storage[binVariable] )
            
        if loggedValues:
            valuesToBeBinned = np.array( np.log10( self.Storage[variableToBeBinned] ) )
        else:
            valuesToBeBinned = np.array( self.Storage[variableToBeBinned] )

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

        # Saving Binned Data to Storage
        binDataStorage['rawBinnedData'] = np.array(binnedDataTemp)
        binDataStorage['binCenters'] = np.array(binCenters)

        del binMask, binCenters, binEdges

        #####################
        # This section will carry out some averaging of the binned data
        # We will calculate the median, the 60-40 percentiles, 75-25 percentiles and the number density

        binDataStorage['Median'] = np.array( [np.median(x[~np.isnan(x)]) for x in binnedDataTemp] )        
        binDataStorage['NumberDensity'] = np.array( [len(x) for x in binnedDataTemp] )
        binDataStorage['STD'] = np.array( [np.std(x[(~np.isnan(x)) & (np.isfinite(x))]) for x in binnedDataTemp] )        
        binDataStorage['Mean'] = np.array( [np.mean(x[(~np.isnan(x)) & (np.isfinite(x))]) for x in binnedDataTemp] ) 
        
        # These percentiles may need protection in place in case they fail but we shall wait and see
        binDataStorage['Percentile25'] = np.array( [np.percentile(x[~np.isnan(x)], 25) for x in binnedDataTemp] ) 
        binDataStorage['Percentile75'] = np.array( [np.percentile(x[~np.isnan(x)], 75) for x in binnedDataTemp] )

        ######################
        # Finally the temp dictionary will be stored in the main storage dictionary 
        self.binStorage["{} vs {}".format(binVariable, variableToBeBinned)] = binDataStorage

        del binDataStorage, binnedDataTemp

    
    
    def DifferenceValues(self, name, key, perGal = False, Return = False):
        '''
        '''
        # Single Run
        # Converting to array if only one is required
        
        if type(key) == str:
            key = [key]

        if type(name) == str:
            name = [name]
            
        
        if not perGal:

            # Looping over Keys
            for k in key:
                if not Return: 
                    self.Storage[name] = np.array([(self.Storage[k][i+1] - self.Storage[k][i]) for i in range(len(self.Storage[k]) -1)])
                else:
                    return np.array([(self.Storage[k][i+1] - self.Storage[k][i]) for i in range(len(self.Storage[k]) -1)])
        # Looping over Galaxies
        # Requires TopLeafID in self.Storage
        else:

            for i, k in enumerate(key):
                temp = []
                IDs = np.unique(self.Storage['TopLeafID'])
                print( 'Calculating Difference: {}'.format(k) )
                ts = time.time()
                
                for j,ID in enumerate(IDs):
                    maskID = self.Storage['TopLeafID'] == ID
                    data = self.Storage[k][maskID]
#                     diffData = [(data[i+1] - data[i]) for i in range(len(data) -1)]
                    diffData = np.diff(data)
                    temp = np.concatenate((temp, diffData))

                    if j == 100:
                        t = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

                        sortTime = (time.time() - ts) / 100
                        log = '{} | Estimated Time: {}s'.format(t,round(sortTime*len(IDs),2))
                        print(log)
                if not Return: 
                    self.Storage[name[i]] = temp
                else:
                    return temp
        
    def AverageValues(self, name, key, perGal = False, Return = False):
        '''This Function takes one of the 
        '''
        # Single Run
        # Converting to array if only one is required  
        if type(key) == str:
            key = [key]
        if type(name) == str:
            name = [name]
        
        if not perGal:
            # Looping over Keys
            for k in key:
                if not Return: 
                    self.Storage[name] = np.array([(self.Storage[k][i+1] + self.Storage[k][i])/2 for i in range(len(self.Storage[k]) -1)])
                else:
                    return np.array([(self.Storage[k][i+1] + self.Storage[k][i])/2 for i in range(len(self.Storage[k]) -1)])
        # Looping over Galaxies
        # Requires TopLeafID in self.Storage
        else:

            for i, k in enumerate(key):
                temp = []
                IDs = np.unique(self.Storage['TopLeafID'])
                print( 'Calculating Average: {}'.format(k) )
                ts = time.time()
                
                for j,ID in enumerate(IDs):
                    maskID = self.Storage['TopLeafID'] == ID
                    data = self.Storage[k][maskID]
                    avData = [(data[i+1] + data[i])/2 for i in range(len(data) -1)]
                    temp = np.concatenate((temp, avData))
                    
                    if j == 100:
                        sortTime = (time.time() - ts) / 100
                        t = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                        log = '{} | Estimated Time: {}s'.format(t,round(sortTime*len(IDs),2))
                        print(log)

                if not Return: 
                    self.Storage[name[i]] = temp
                else:
                    return temp
        
    def RateValues(self, name, key, ratekey, perGal = False):
        '''
        '''
        if type(key) == str:
            key = [key]

        if type(name) == str:
            name = [name]

            
        dataDen = self.DifferenceValues(' ',  ratekey, perGal = perGal, Return = True) *1e9

        for i, k in enumerate(key):
            dataNum = self.DifferenceValues(' ',  k, perGal = perGal, Return = True)
            ratio = dataNum / dataDen
            self.Storage[name[i]] = ratio

    
    def FitModel(self, model, x, y):
        # Fitting the Model to the Data
        popt , pcov = curve_fit(model, x, y)
        params = popt
        errors = np.sqrt(np.diag(pcov))   
        ymodel = model(x, *params)

        DoF = len(x) - len(params)
        chiSq = round(self.ChiSq(y,ymodel), 1 - int(math.floor(math.log10(abs(self.ChiSq(y,ymodel))))) - 1)
        rChiSq = round(chiSq/DoF, 1 - int(math.floor(math.log10(abs(chiSq/DoF)))) - 1)
        a,b = unc.correlated_values(popt, pcov)

        xp = np.linspace(min(x), max(x) , 1000)
        yp = a * xp + b

        nom = unp.nominal_values(yp)
        std = unp.std_devs(yp)

        # Calculating CI
        lpb , upb = self.predband(xp, x , y, params, model, conf = 0.95)

        print('ChiSquared: {} \n Reduced ChiSquare: {} \n Fitted Parameters: {}, {} \n Parameter Errors: {}, {}'.format(chiSq, rChiSq, *np.round(params,2), *np.round(errors,2) ))
        print('---------------------------------')
        
        confidence95 = [xp, nom - 1.96 * std, nom + 1.96 * std]
        confidenceInterval = [xp, lpb, upb]
        fit = [xp, ymodel, params]
        return fit, confidence95, confidenceInterval
        
    
    
    def ChiSq(self, y, ymodel ):
        return np.sum( ((y - ymodel)**2) / abs(ymodel) )

    def predband(self, x, xd, yd, p, func, conf=0.95):
        # x = requested points
        # xd = x data
        # yd = y data
        # p = parameters
        # func = function name
        alpha = 1.0 - conf    # significance
        N = xd.size          # data sample size
        var_n = len(p)  # number of parameters
        # Quantile of Student's t distribution for p=(1-alpha/2)
        q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
        # Stdev of an individual measurement
        se = np.sqrt(1. / (N - var_n) * \
                     np.sum((yd - func(xd, *p)) ** 2))
        # Auxiliary definitions
        sx = (x - xd.mean()) ** 2
        sxd = np.sum((xd - xd.mean()) ** 2)
        # Predicted values (best-fit model)
        yp = func(x, *p)
        # Prediction band
        dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
        # Upper & lower prediction bands.
        lpb, upb = yp - dy, yp + dy
        return lpb, upb
    
    
    def RedshiftBinning(self, x, y, SNrange, numBins, binRanges, AverageSnapshot = False, loggedBins = False, loggedValues = False):

        for SN, binRange in zip(SNrange, binRanges):
            labx = "{}| {}".format(SN, x)
            laby = "{}| {}".format(SN, y)
            
            if AverageSnapshot:
                mask = (self.Storage['avSN'] == SN)
            else:
                mask = (self.Storage['SnapShot'] == SN)    
                
            self.Storage[labx] = self.Storage[x][mask]
            self.Storage[laby] = self.Storage[y][mask]
        
            print('{}| Minimum: {},   Maximum: {},   Length: {}'.format(SN, np.log10(min(self.Storage[labx])), np.log10(max(self.Storage[labx])), len(self.Storage[labx])))
            self.Binning(numBins, labx, laby, binRange, loggedBins = loggedBins, loggedValues = loggedValues)
            del self.Storage[labx], self.Storage[laby]
        