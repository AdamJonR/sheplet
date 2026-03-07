use conversations::*;

fn temp_store() -> (ConversationStore, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let store = ConversationStore::open(dir.path().join("db")).unwrap();
    (store, dir)
}

#[test]
fn create_and_get_roundtrip() {
    let (store, _dir) = temp_store();
    let conv = store.create_conversation("bio101", "Test Chat").unwrap();
    assert_eq!(conv.course_id, "bio101");
    assert_eq!(conv.title, "Test Chat");
    assert!(conv.messages.is_empty());

    let fetched = store.get(&conv.id).unwrap().unwrap();
    assert_eq!(fetched.id, conv.id);
    assert_eq!(fetched.course_id, "bio101");
    assert_eq!(fetched.title, "Test Chat");
}

#[test]
fn append_messages_and_verify_order() {
    let (store, _dir) = temp_store();
    let conv = store.create_conversation("chem201", "Reactions").unwrap();

    store
        .append_message(
            &conv.id,
            Message {
                role: Role::User,
                content: "What is oxidation?".to_string(),
                timestamp: "2026-01-01T00:00:00Z".to_string(),
                citations: vec![],
            },
        )
        .unwrap();

    store
        .append_message(
            &conv.id,
            Message {
                role: Role::Assistant,
                content: "Oxidation is the loss of electrons.".to_string(),
                timestamp: "2026-01-01T00:00:01Z".to_string(),
                citations: vec![Citation {
                    source_file: "ch5.pdf".to_string(),
                    chunk_index: 3,
                    text_snippet: "Oxidation involves...".to_string(),
                }],
            },
        )
        .unwrap();

    let fetched = store.get(&conv.id).unwrap().unwrap();
    assert_eq!(fetched.messages.len(), 2);
    assert_eq!(fetched.messages[0].role, Role::User);
    assert_eq!(fetched.messages[1].role, Role::Assistant);
    assert_eq!(fetched.messages[1].citations.len(), 1);
    assert_eq!(fetched.messages[1].citations[0].source_file, "ch5.pdf");
}

#[test]
fn list_by_course_prefix_scan() {
    let (store, _dir) = temp_store();
    store.create_conversation("bio101", "Chat 1").unwrap();
    store.create_conversation("bio101", "Chat 2").unwrap();
    store.create_conversation("chem201", "Chat 3").unwrap();

    let bio = store.list_by_course("bio101").unwrap();
    assert_eq!(bio.len(), 2);
    assert!(bio.iter().all(|s| s.course_id == "bio101"));

    let chem = store.list_by_course("chem201").unwrap();
    assert_eq!(chem.len(), 1);

    let all = store.list_all().unwrap();
    assert_eq!(all.len(), 3);
}

#[test]
fn delete_individual_conversation() {
    let (store, _dir) = temp_store();
    let conv1 = store.create_conversation("bio101", "Chat 1").unwrap();
    let conv2 = store.create_conversation("bio101", "Chat 2").unwrap();

    store.delete(&conv1.id).unwrap();

    assert!(store.get(&conv1.id).unwrap().is_none());
    assert!(store.get(&conv2.id).unwrap().is_some());
    assert_eq!(store.list_by_course("bio101").unwrap().len(), 1);
}

#[test]
fn clear_course_conversations() {
    let (store, _dir) = temp_store();
    store.create_conversation("bio101", "Chat 1").unwrap();
    store.create_conversation("bio101", "Chat 2").unwrap();
    store.create_conversation("chem201", "Chat 3").unwrap();

    store.clear_course("bio101").unwrap();

    assert_eq!(store.list_by_course("bio101").unwrap().len(), 0);
    assert_eq!(store.list_by_course("chem201").unwrap().len(), 1);
}

#[test]
fn empty_store_returns_empty_lists() {
    let (store, _dir) = temp_store();
    assert!(store.list_all().unwrap().is_empty());
    assert!(store.list_by_course("nonexistent").unwrap().is_empty());
    assert!(store.get("nonexistent").unwrap().is_none());
}

#[test]
fn export_formatting() {
    let (store, _dir) = temp_store();
    let conv = store.create_conversation("bio101", "Mitosis Q&A").unwrap();

    store
        .append_message(
            &conv.id,
            Message {
                role: Role::User,
                content: "What is mitosis?".to_string(),
                timestamp: "2026-01-01T10:00:00Z".to_string(),
                citations: vec![],
            },
        )
        .unwrap();

    store
        .append_message(
            &conv.id,
            Message {
                role: Role::Assistant,
                content: "Mitosis is cell division.".to_string(),
                timestamp: "2026-01-01T10:00:01Z".to_string(),
                citations: vec![Citation {
                    source_file: "ch3.pdf".to_string(),
                    chunk_index: 12,
                    text_snippet: "Mitosis is the process...".to_string(),
                }],
            },
        )
        .unwrap();

    let conv = store.get(&conv.id).unwrap().unwrap();
    let txt = export_as_txt(&conv);

    assert!(txt.contains("Conversation: Mitosis Q&A"));
    assert!(txt.contains("Course: bio101"));
    assert!(txt.contains("Student:"));
    assert!(txt.contains("What is mitosis?"));
    assert!(txt.contains("Tutor:"));
    assert!(txt.contains("Mitosis is cell division."));
    assert!(txt.contains("ch3.pdf (chunk 12)"));
}

#[test]
fn summary_has_correct_message_count() {
    let (store, _dir) = temp_store();
    let conv = store.create_conversation("bio101", "Test").unwrap();

    store
        .append_message(
            &conv.id,
            Message {
                role: Role::User,
                content: "Hello".to_string(),
                timestamp: "2026-01-01T00:00:00Z".to_string(),
                citations: vec![],
            },
        )
        .unwrap();

    let summaries = store.list_by_course("bio101").unwrap();
    assert_eq!(summaries[0].message_count, 1);
}
