USE [Home_Credit]
GO

DROP TABLE IF EXISTS [dbo].[Bureau_Balance_STG]
GO

CREATE TABLE [dbo].[Bureau_Balance_STG](
SK_ID_BUREAU	nvarchar(500) NULL,
MONTHS_BALANCE	nvarchar(500) NULL,
STATUS	nvarchar(500) NULL
)