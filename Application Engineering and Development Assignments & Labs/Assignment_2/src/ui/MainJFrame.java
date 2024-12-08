/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/GUIForms/JFrame.java to edit this template
 */
package ui;

import java.awt.CardLayout;
import java.awt.Color;
import javax.swing.JOptionPane;
import model.Address;
import model.Person;
import model.PersonDirectory;

/**
 *
 * @author tawde
 */
public class MainJFrame extends javax.swing.JFrame {
    private PersonDirectory personDirectory;
    private Person person1;
    private Person person2;
    private Person person3;
    private Person person4;
    private Person person5;
    
    private Address workAddress1;
    private Address homeAddress1;
    
    private Address workAddress2;
    private Address homeAddress2;
    
    private Address workAddress3;
    private Address homeAddress3;
    
    private Address workAddress4;
    private Address homeAddress4;
    
    private Address workAddress5;
    private Address homeAddress5;

    /**
     * Creates new form MainJFrame
     */
    public MainJFrame() {
        initComponents();
        this.personDirectory = new PersonDirectory();
        this.workAddress1 = new Address();
        this.homeAddress1 = new Address();
        this.workAddress2 = new Address();
        this.homeAddress2 = new Address();
        this.workAddress3 = new Address();
        this.homeAddress3 = new Address();
        this.workAddress4 = new Address();
        this.homeAddress4 = new Address();
        this.workAddress5 = new Address();
        this.homeAddress5 = new Address();
        
        Person person1 = personDirectory.addPerson();
        person1.setFirstName("Marcus");
        person1.setLastName("Rashford");
        person1.setSsn(78256);
        person1.setAge((byte) 26);

        workAddress1.setStreetAddress("47 Iffley Road");
        workAddress1.setUnitNumber("3");
        workAddress1.setCity("Boston");
        workAddress1.setState("MA");
        workAddress1.setZipCode(02130);
        person1.setWorkAddress(workAddress1);
        
        homeAddress1.setStreetAddress("360 street");
        homeAddress1.setUnitNumber("10");
        homeAddress1.setCity("Boston");
        homeAddress1.setState("Massachusetts");
        homeAddress1.setZipCode(21150);
        person1.setHomeAddress(homeAddress1);
        
        Person person2 = personDirectory.addPerson();
        person2.setFirstName("Bruno");
        person2.setLastName("Fernandes");
        person2.setSsn(94348);
        person2.setAge((byte) 30);

        workAddress2.setStreetAddress("30 Weld Street Road");
        workAddress2.setUnitNumber("5");
        workAddress2.setCity("Boston");
        workAddress2.setState("MA");
        workAddress2.setZipCode(02150);
        person2.setWorkAddress(workAddress2);
        
        homeAddress2.setStreetAddress("760 street");
        homeAddress2.setUnitNumber("15");
        homeAddress2.setCity("Boston");
        homeAddress2.setState("Massachusetts");
        homeAddress2.setZipCode(21760);
        person2.setHomeAddress(homeAddress2);
        
        Person person3 = personDirectory.addPerson();
        person3.setFirstName("Joshua");
        person3.setLastName("Zirkzee");
        person3.setSsn(49676);
        person3.setAge((byte) 23);

        workAddress3.setStreetAddress("25 Walnut Street Road");
        workAddress3.setUnitNumber("7");
        workAddress3.setCity("Boston");
        workAddress3.setState("MA");
        workAddress3.setZipCode(02110);
        person3.setWorkAddress(workAddress3);
        
        homeAddress3.setStreetAddress("890 street");
        homeAddress3.setUnitNumber("18");
        homeAddress3.setCity("Boston");
        homeAddress3.setState("Massachusetts");
        homeAddress3.setZipCode(21460);
        person3.setHomeAddress(homeAddress3);
        
        Person person4 = personDirectory.addPerson();
        person4.setFirstName("Alejandro");
        person4.setLastName("Garnacho");
        person4.setSsn(45876);
        person4.setAge((byte) 20);

        workAddress4.setStreetAddress("27 Roxbury Road");
        workAddress4.setUnitNumber("77");
        workAddress4.setCity("Boston");
        workAddress4.setState("MA");
        workAddress4.setZipCode(02140);
        person4.setWorkAddress(workAddress4);
        
        homeAddress4.setStreetAddress("892 street");
        homeAddress4.setUnitNumber("20");
        homeAddress4.setCity("Boston");
        homeAddress4.setState("Massachusetts");
        homeAddress4.setZipCode(21490);
        person4.setHomeAddress(homeAddress4);
        
        Person person5 = personDirectory.addPerson();
        person5.setFirstName("Mason");
        person5.setLastName("Mount");
        person5.setSsn(45116);
        person5.setAge((byte) 25);

        workAddress5.setStreetAddress("30 Centre Street Road");
        workAddress5.setUnitNumber("97");
        workAddress5.setCity("Boston");
        workAddress5.setState("MA");
        workAddress5.setZipCode(02120);
        person5.setWorkAddress(workAddress5);
        
        homeAddress5.setStreetAddress("895 street");
        homeAddress5.setUnitNumber("20");
        homeAddress5.setCity("Boston");
        homeAddress5.setState("Massachusetts");
        homeAddress5.setZipCode(21890);
        person5.setHomeAddress(homeAddress5);
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jSplitPane1 = new javax.swing.JSplitPane();
        topJPanel = new javax.swing.JPanel();
        btnAddPerson = new javax.swing.JButton();
        btnListPerson = new javax.swing.JButton();
        jLabel1 = new javax.swing.JLabel();
        btnSearchPerson = new javax.swing.JButton();
        txtSearchInput = new javax.swing.JTextField();
        userProcessContainer = new javax.swing.JPanel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        topJPanel.setBackground(new java.awt.Color(0, 51, 102));

        btnAddPerson.setBackground(new java.awt.Color(204, 102, 0));
        btnAddPerson.setForeground(new java.awt.Color(255, 255, 255));
        btnAddPerson.setText("Add Person");
        btnAddPerson.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnAddPersonActionPerformed(evt);
            }
        });

        btnListPerson.setBackground(new java.awt.Color(204, 102, 0));
        btnListPerson.setForeground(new java.awt.Color(255, 255, 255));
        btnListPerson.setText("List Person");
        btnListPerson.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnListPersonActionPerformed(evt);
            }
        });

        jLabel1.setFont(new java.awt.Font("Segoe UI", 0, 18)); // NOI18N
        jLabel1.setForeground(new java.awt.Color(255, 255, 255));
        jLabel1.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        jLabel1.setText("Person Profile");

        btnSearchPerson.setBackground(new java.awt.Color(204, 102, 0));
        btnSearchPerson.setForeground(new java.awt.Color(255, 255, 255));
        btnSearchPerson.setText("Seach for Person");
        btnSearchPerson.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnSearchPersonActionPerformed(evt);
            }
        });

        txtSearchInput.setForeground(new java.awt.Color(102, 102, 102));
        txtSearchInput.setText("Type name or street address");
        txtSearchInput.addFocusListener(new java.awt.event.FocusAdapter() {
            public void focusGained(java.awt.event.FocusEvent evt) {
                txtSearchInputFocusGained(evt);
            }
            public void focusLost(java.awt.event.FocusEvent evt) {
                txtSearchInputFocusLost(evt);
            }
        });
        txtSearchInput.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                txtSearchInputActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout topJPanelLayout = new javax.swing.GroupLayout(topJPanel);
        topJPanel.setLayout(topJPanelLayout);
        topJPanelLayout.setHorizontalGroup(
            topJPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jLabel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addGroup(topJPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(topJPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(btnSearchPerson, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(btnListPerson, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(btnAddPerson, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
            .addGroup(topJPanelLayout.createSequentialGroup()
                .addComponent(txtSearchInput, javax.swing.GroupLayout.PREFERRED_SIZE, 167, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 0, Short.MAX_VALUE))
        );
        topJPanelLayout.setVerticalGroup(
            topJPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(topJPanelLayout.createSequentialGroup()
                .addGap(25, 25, 25)
                .addComponent(jLabel1)
                .addGap(96, 96, 96)
                .addComponent(btnAddPerson)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(btnListPerson)
                .addGap(103, 103, 103)
                .addComponent(txtSearchInput, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addComponent(btnSearchPerson)
                .addContainerGap(380, Short.MAX_VALUE))
        );

        jSplitPane1.setLeftComponent(topJPanel);

        userProcessContainer.setBackground(new java.awt.Color(0, 51, 102));
        userProcessContainer.setLayout(new java.awt.CardLayout());
        jSplitPane1.setRightComponent(userProcessContainer);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jSplitPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 859, Short.MAX_VALUE)
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jSplitPane1, javax.swing.GroupLayout.Alignment.TRAILING)
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void btnAddPersonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnAddPersonActionPerformed
        // TODO add your handling code here:
        CreatePersonJPanel panel = new CreatePersonJPanel(userProcessContainer, personDirectory);
        userProcessContainer.add("CreatePersonJPanel", panel);
        CardLayout layout = (CardLayout) userProcessContainer.getLayout();
        layout.next(userProcessContainer);
    }//GEN-LAST:event_btnAddPersonActionPerformed

    private void btnSearchPersonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnSearchPersonActionPerformed
        // TODO add your handling code here:
        Person result = personDirectory.searchPerson(txtSearchInput.getText());
        
        if("".equals(txtSearchInput.getText())) {
            JOptionPane.showMessageDialog(null, "Please enter a first name/last name or street address!", "Warning", JOptionPane.WARNING_MESSAGE);
        }
        else {
            if(result == null) {
                JOptionPane.showMessageDialog(null, "Person does not exist!", "Information", JOptionPane.INFORMATION_MESSAGE);
            }
            else {
                Address workResult = result.getWorkAddress();
                Address homeResult = result.getHomeAddress();
        
                ViewPersonJPanel panel = new ViewPersonJPanel(userProcessContainer, result, workResult, homeResult);
                userProcessContainer.add("ViewPersonJPanel", panel);
                CardLayout layout = (CardLayout) userProcessContainer.getLayout();
                layout.next(userProcessContainer);
            }
        }
    }//GEN-LAST:event_btnSearchPersonActionPerformed

    private void btnListPersonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnListPersonActionPerformed
        // TODO add your handling code here:
        ManagePersonJPanel panel = new ManagePersonJPanel(userProcessContainer, personDirectory);
        userProcessContainer.add("ManagePersonJPanel", panel);
        CardLayout layout = (CardLayout) userProcessContainer.getLayout();
        layout.next(userProcessContainer);
    }//GEN-LAST:event_btnListPersonActionPerformed

    private void txtSearchInputFocusGained(java.awt.event.FocusEvent evt) {//GEN-FIRST:event_txtSearchInputFocusGained
        // TODO add your handling code here:
        if(txtSearchInput.getText().equalsIgnoreCase("Type name or street address"))
        {
            txtSearchInput.setText("");
            txtSearchInput.setForeground(new Color(0,0,0));
        }
    }//GEN-LAST:event_txtSearchInputFocusGained

    private void txtSearchInputFocusLost(java.awt.event.FocusEvent evt) {//GEN-FIRST:event_txtSearchInputFocusLost
        // TODO add your handling code here:
        if(txtSearchInput.getText().equalsIgnoreCase(""))
        {
            txtSearchInput.setText("Type name or street address");
            txtSearchInput.setForeground(new Color(0,0,0));
        }

    }//GEN-LAST:event_txtSearchInputFocusLost

    private void txtSearchInputActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_txtSearchInputActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_txtSearchInputActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(MainJFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(MainJFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(MainJFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(MainJFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new MainJFrame().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton btnAddPerson;
    private javax.swing.JButton btnListPerson;
    private javax.swing.JButton btnSearchPerson;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JSplitPane jSplitPane1;
    private javax.swing.JPanel topJPanel;
    private javax.swing.JTextField txtSearchInput;
    private javax.swing.JPanel userProcessContainer;
    // End of variables declaration//GEN-END:variables
}
