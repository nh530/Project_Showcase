USE [Home_Credit]
GO

DROP TABLE IF EXISTS [dbo].[Installments_Payments_STG]
GO

CREATE TABLE [dbo].[Installments_Payments_STG](
	[SK_ID_PREV] [nvarchar](500) NULL,
	[SK_ID_CURR] [nvarchar](500) NULL,
	[NUM_INSTALMENT_VERSION] [nvarchar](500) NULL,
	[NUM_INSTALMENT_NUMBER] [nvarchar](500) NULL,
	[DAYS_INSTALMENT] [nvarchar](500) NULL,
	[DAYS_ENTRY_PAYMENT] [nvarchar](500) NULL,
	[AMT_INSTALMENT] [nvarchar](500) NULL,
	[AMT_PAYMENT] [nvarchar](500) NULL
) ON [PRIMARY]
GO


