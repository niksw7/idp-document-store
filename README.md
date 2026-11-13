# idp-document-store

Document management system (DMS) for the IDP agent demo. Documents live
under `documents/` as `<document_id>.pdf`. The IDP agent references
documents by id only and fetches the bytes itself
(`https://raw.githubusercontent.com/niksw7/idp-document-store/main/documents/<id>.pdf`),
so no document bytes ever cross the request boundary.

Synthetic KYC fixture data for a backoffice agent scrutiny demo.
