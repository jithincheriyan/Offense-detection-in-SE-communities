


politeness_function<-function(comment)
  {
  options(warn=1)
  #data=read.csv(file_path)
  #politeness::politeness(data$Comment_Text)
  #spacyr::spacy_initialize(python_executable = "C:\\Users\\cheji902\\PycharmProjects\\Paraphrase\\Scripts\\spacy.exe")
  #comment="hello dear"
  out=data.frame(politeness::politeness(comment))
  out
  return(out)
  #write_delim(out, "C:\\StackOverflow_Work\\UN_Test_Data_Politeness.csv", delim=",", na="?")
}






#PYTHON_PATH="C:/Users/cheji902/AppData/Local/Programs/Python/Python37"
# install.packages("spacyr")
#spacyr::spacy_initialize(python_executable = PYTHON_PATH)
#politeness::politeness(phone_offers$message, parser="spacy")

#politeness::politenessPlot(politeness::politeness(data$Comment_Text),
                           #split=phone_offers$condition,
                           #split_levels = c("Warm","Tough"),
                           #split_name = "Condition")





# fpt_most<-findPoliteTexts(comment,
#                           df_polite_train,
#                           phone_offers$condition,
#                           type="most")
# fpt_least<-findPoliteTexts(phone_offers$message,
#                            df_polite_train,
#                            phone_offers$condition,
#                            type="least")
