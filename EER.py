def getErrorRate(scoreList, threshold=0):

    TRNum = 0    
    FRNum = 0    
    FANum = 0    
    TANum = 0    
    
    for element in scoreList:
        featureType, logLikelihoodRatio = element
        
        if logLikelihoodRatio < threshold:
            if featureType == 'trueSpeaker':
                FRNum += 1
            elif featureType == 'imposter':
                TRNum += 1
       
        else:        
            if featureType == 'trueSpeaker':
                TANum += 1
            elif featureType == 'imposter':
                FANum += 1
                
    imposterNum = TRNum + FANum
    trueSpeakerNum = FRNum + TANum
    
    FARate = float(FANum) / float(imposterNum) * 100.        
    FRRate = float(FRNum) / float(trueSpeakerNum) * 100.     
    
    correctNum = TRNum + TANum
    wrongNum = FANum + FRNum
    
    return FARate, FRRate, correctNum, wrongNum


def calculateEER(scoreList):
    boundary = 0.001            
    repeatNum = 500                
    
    left = -100.
    right = 100.
    
    
    for index in range(repeatNum): #@UnusedVariable
                    
        
        middle = (left + right) / 2.0
        
        FARate, FRRate, correctNum, wrongNum = getErrorRate(scoreList, threshold=middle)
        errorRate = FRRate - FARate
        
        if abs(errorRate) <= boundary:
            return middle
        
        
        if errorRate < 0:
            left = middle
        else:
            right = middle
            
    return middle
        